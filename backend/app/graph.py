"""
graph.py - Neo4j entity graph storage.
"""
from __future__ import annotations

import json
import logging
import hashlib
import re
from typing import Protocol

from app.config import get_settings
from app.models import Chunk, EntityMention, GraphEntity, GraphRelationship

logger = logging.getLogger(__name__)

_NUMERIC_ENTITY_LABELS = {
    "REVENUE",
    "PROFIT",
    "LOSS",
    "EXPENSE",
    "CASHFLOW",
    "PAYMENT",
    "TRANSACTION",
    "ASSET",
    "LIABILITY",
    "EQUITY",
}
_NUMERIC_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}
_NUMERIC_VALUE_RE = re.compile(
    r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(k|m|b|thousand|million|billion|trillion)?",
    re.IGNORECASE,
)


class GraphStore(Protocol):
    def upsert_document_graph(self, document: str, chunks: list[Chunk], mentions: list[EntityMention]) -> int: ...
    def search_entities(self, query: str, limit: int = 10) -> list[GraphEntity]: ...
    def neighbors(self, entity_id: str, depth: int = 1, limit: int = 50) -> tuple[GraphEntity | None, list[GraphEntity], list[GraphRelationship]]: ...
    def chunks_for_entities(self, entity_ids: list[str], limit: int = 20) -> list[tuple[str, int, int, list[str]]]: ...
    def relationship_facts(self, relationship_types: list[str] | None = None, source_id: str | None = None, target_id: str | None = None, limit: int = 100) -> list[dict]: ...
    def delete_document(self, document: str) -> None: ...


_RELATION_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "by", "of", "to", "for", "from", "with", "in", "on", "at", "as",
    "and", "or", "all", "every", "list", "show", "give", "display",
    "enumerate", "their", "its", "his", "her",
}
_RELATION_QUERY_RE = re.compile(
    r"\b(who|what|which)\s+(?P<relation>[a-z][a-z\s]{1,60}?)\s+"
    r"(?:(?P<label>[A-Z][a-z][A-Za-z]{1,30})\s+)?(?P<id>[A-Z]{1,10}-\d+)\b",
    re.IGNORECASE,
)
_ENTITY_PREFIX_RE = re.compile(
    r"^(?:company|corporation|corp\.?|subsidiary|investor|shareholder|"
    r"founder|ceo|executive|board\s+member|person|billionaire|entrepreneur|"
    r"project|product|business\s+unit|contract|agreement|partnership|"
    r"invoice|payment|transaction|revenue|profit|loss|expense|cashflow|"
    r"asset|liability|equity|stock|bond|fund|etf|bank|financial\s+institution|"
    r"department|employee|country|city|region|quarter|fiscal\s+year|date)\s+",
    re.IGNORECASE,
)


def _entity_id(label: str, normalized: str) -> str:
    digest = hashlib.sha1(f"{label}:{normalized}".encode("utf-8")).hexdigest()[:12]
    return f"{label.lower()}:{digest}"


def _label_for_prefixed_id(noun: str | None, entity_text: str) -> str:
    if noun:
        return f"{noun.upper()}_ID"
    prefix = entity_text.split("-", 1)[0].upper()
    return f"{prefix}_ID"


def _normalize_entity(text: str, label: str) -> str:
    value = re.sub(r"\s+", " ", text.strip())
    value = _ENTITY_PREFIX_RE.sub("", value).strip()
    if label.endswith("_ID") or label in {"ID", "EMAIL", "PHONE"}:
        return value.lower()
    return re.sub(r"[^\w\s&.-]", "", value.lower()).strip()


def _stem_relation_word(word: str) -> str:
    if word.endswith("ers") and len(word) > 5:
        return word[:-3]
    if word.endswith("er") and len(word) > 4:
        return word[:-2]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _normalize_relation(text: str) -> str | None:
    words = re.findall(r"[a-z]+", text.lower())
    words = [
        _stem_relation_word(word)
        for word in words
        if word not in _RELATION_STOPWORDS
    ]
    if not words:
        return None
    return "_".join(words[:5]).upper()


def _numeric_value(text: str, label: str) -> float | None:
    if label not in _NUMERIC_ENTITY_LABELS:
        return None
    match = _NUMERIC_VALUE_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (match.group(2) or "").lower()
    return value * _NUMERIC_MULTIPLIERS.get(unit, 1)


def _relation_tokens(text: str) -> set[str]:
    normalized = _normalize_relation(text) or ""
    return {token for token in normalized.lower().split("_") if token}


def graph_entity_id(label: str, text: str) -> str:
    return _entity_id(label, _normalize_entity(text, label))


def graph_label_for_id(noun: str | None, entity_text: str) -> str:
    return _label_for_prefixed_id(noun, entity_text)


def graph_relation_tokens(text: str) -> set[str]:
    return _relation_tokens(text)


def create_graph_store() -> GraphStore:
    settings = get_settings()
    if settings.graph_backend.lower() != "neo4j":
        raise ValueError(
            f"Unsupported GRAPH_BACKEND={settings.graph_backend!r}. "
            "This application is configured for Neo4j only."
        )
    return Neo4jGraphStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )


def _relationships_from_llm_extractions(
    document: str,
    chunks: list[Chunk],
    mentions: list[EntityMention],
) -> list[GraphRelationship]:
    by_chunk: dict[tuple[int, int], list[EntityMention]] = {}
    for mention in mentions:
        by_chunk.setdefault((mention.page or 0, mention.chunk_index or 0), []).append(mention)

    rels: dict[tuple[str, str, str], GraphRelationship] = {}
    for chunk in chunks:
        chunk_mentions = by_chunk.get((chunk.page, chunk.chunk_index), [])
        mention_index = _mention_index(chunk_mentions)
        for relation in chunk.kg_relations:
            source_name = str(relation.get("source") or "").strip()
            relation_type = str(relation.get("relation") or "").strip().upper()
            target_name = str(relation.get("target") or "").strip()
            source = _find_mention(mention_index, source_name)
            target = _find_mention(mention_index, target_name)
            if not source or not target or not relation_type:
                continue
            evidence = {
                "document": document,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "source_label": source.label,
                "source_name": source.text,
                "target_label": target.label,
                "target_name": target.text,
                "relation_text": relation.get("relation"),
            }
            rels[(source.entity_id, target.entity_id, relation_type)] = GraphRelationship(
                source_id=source.entity_id,
                target_id=target.entity_id,
                type=relation_type,
                confidence=0.92,
                evidence=evidence,
            )

    return list(rels.values())


def _mention_index(mentions: list[EntityMention]) -> dict[str, EntityMention]:
    index: dict[str, EntityMention] = {}
    for mention in mentions:
        index.setdefault(mention.text.strip().lower(), mention)
        index.setdefault(mention.normalized.strip().lower(), mention)
    return index


def _find_mention(
    mention_index: dict[str, EntityMention],
    name: str,
) -> EntityMention | None:
    key = name.strip().lower()
    if key in mention_index:
        return mention_index[key]
    normalized = _normalize_entity(name, "ENTITY")
    return mention_index.get(normalized)


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        from neo4j import GraphDatabase  # type: ignore

        if not uri.startswith("neo4j+s://"):
            raise ValueError("NEO4J_URI must use neo4j+s:// for Neo4j Aura")
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        with self._driver.session(database=self._database) as session:
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
            session.run("CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE")
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")

    def upsert_document_graph(self, document: str, chunks: list[Chunk], mentions: list[EntityMention]) -> int:
        relationships = _relationships_from_llm_extractions(document, chunks, mentions)
        entity_count = len({mention.entity_id for mention in mentions})
        mention_count = len(mentions)
        relationship_count = len(relationships)
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (d:Document {name: $document})-[r:MENTIONS]->() DELETE r", document=document)
            session.run(
                "MATCH ()-[r:RELATED {document: $document}]->() DELETE r",
                document=document,
            )
            session.run("MERGE (:Document {name: $document})", document=document)
            for mention in mentions:
                numeric_value = _numeric_value(mention.text, mention.label)
                session.run(
                    """
                    MERGE (e:Entity {entity_id: $entity_id})
                    SET e.label = $label, e.name = $name,
                        e.numeric_value = CASE WHEN $numeric_value IS NULL THEN e.numeric_value ELSE $numeric_value END,
                        e.confidence = CASE WHEN e.confidence IS NULL OR e.confidence < $confidence THEN $confidence ELSE e.confidence END,
                        e.aliases = CASE
                            WHEN e.aliases IS NULL THEN [$alias]
                            WHEN NOT ($alias IN e.aliases) THEN e.aliases + $alias
                            ELSE e.aliases
                        END
                    WITH e
                    MATCH (d:Document {name: $document})
                    MERGE (d)-[m:MENTIONS {page: $page, chunk_index: $chunk_index, text: $text}]->(e)
                    SET m.confidence = $confidence
                    """,
                    entity_id=mention.entity_id,
                    label=mention.label,
                    name=mention.normalized,
                    alias=mention.text,
                    numeric_value=numeric_value,
                    confidence=mention.confidence,
                    document=document,
                    page=mention.page or 0,
                    chunk_index=mention.chunk_index or 0,
                    text=mention.text,
                )
            for rel in relationships:
                session.run(
                    """
                    MERGE (a:Entity {entity_id: $source_id})
                    MERGE (b:Entity {entity_id: $target_id})
                    MERGE (a)-[r:RELATED {type: $type, document: $document}]->(b)
                    SET r.confidence = $confidence, r.evidence = $evidence, r.document = $document
                    """,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    type=rel.type,
                    confidence=rel.confidence,
                    evidence=json.dumps(rel.evidence),
                    document=document,
                )
        logger.info(
            "Neo4j inserted: %d entities, %d relationships (%d mentions) for %s",
            entity_count,
            relationship_count,
            mention_count,
            document,
        )
        return relationship_count

    def search_entities(self, query: str, limit: int = 10) -> list[GraphEntity]:
        query = _ENTITY_PREFIX_RE.sub("", re.sub(r"\s+", " ", query.strip())).strip()
        if not query:
            return []
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                   OR any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) CONTAINS toLower($query))
                RETURN e
                ORDER BY e.confidence DESC
                LIMIT $limit
                """,
                query=query,
                limit=limit,
            )
            return [self._entity(record["e"]) for record in rows]

    def neighbors(self, entity_id: str, depth: int = 1, limit: int = 50) -> tuple[GraphEntity | None, list[GraphEntity], list[GraphRelationship]]:
        depth = max(1, min(depth, 3))
        rel_pattern = f"*1..{depth}"
        with self._driver.session(database=self._database) as session:
            root = session.run("MATCH (e:Entity {entity_id: $id}) RETURN e", id=entity_id).single()
            neighbor_rows = session.run(
                f"""
                MATCH (e:Entity {{entity_id: $id}})-[:RELATED{rel_pattern}]-(n:Entity)
                RETURN DISTINCT n
                LIMIT $limit
                """,
                id=entity_id,
                limit=limit,
            )
            rel_rows = session.run(
                f"""
                MATCH p=(e:Entity {{entity_id: $id}})-[:RELATED{rel_pattern}]-(n:Entity)
                UNWIND relationships(p) AS r
                RETURN DISTINCT startNode(r).entity_id AS source_id,
                                endNode(r).entity_id AS target_id,
                                r.type AS type,
                                r.confidence AS confidence,
                                r.evidence AS evidence
                LIMIT $limit
                """,
                id=entity_id,
                limit=limit,
            )
            neighbors = [self._entity(record["n"]) for record in neighbor_rows]
            relationships = [
                GraphRelationship(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    type=record["type"],
                    confidence=float(record["confidence"] or 0.0),
                    evidence=json.loads(record["evidence"] or "{}"),
                )
                for record in rel_rows
            ]
        return (self._entity(root["e"]) if root else None, neighbors, relationships)

    def chunks_for_entities(self, entity_ids: list[str], limit: int = 20) -> list[tuple[str, int, int, list[str]]]:
        if not entity_ids:
            return []
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (d:Document)-[m:MENTIONS]->(e:Entity)
                WHERE e.entity_id IN $ids
                WITH d.name AS document, m.page AS page, m.chunk_index AS chunk_index,
                     collect(e.entity_id) AS entities, count(*) AS hits
                RETURN document, page, chunk_index, entities
                ORDER BY hits DESC
                LIMIT $limit
                """,
                ids=entity_ids,
                limit=limit,
            )
            return [(r["document"], int(r["page"]), int(r["chunk_index"]), list(r["entities"])) for r in rows]

    def relationship_facts(
        self,
        relationship_types: list[str] | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        type_filter = "AND r.type IN $types" if relationship_types else ""
        source_filter = "AND a.entity_id = $source_id" if source_id else ""
        target_filter = "AND b.entity_id = $target_id" if target_id else ""
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                f"""
                MATCH (a)-[r:RELATED]->(b)
                WHERE 1 = 1 {type_filter} {source_filter} {target_filter}
                RETURN a, b, r
                ORDER BY r.confidence DESC
                LIMIT $limit
                """,
                types=relationship_types or [],
                source_id=source_id,
                target_id=target_id,
                limit=limit,
            )
            facts = []
            for record in rows:
                source = record["a"]
                target = record["b"]
                rel = record["r"]
                source_aliases = list(source.get("aliases", []) or [])
                target_aliases = list(target.get("aliases", []) or [])
                facts.append({
                    "source_id": source["entity_id"],
                    "target_id": target["entity_id"],
                    "type": rel["type"],
                    "confidence": float(rel.get("confidence", 0.0)),
                    "evidence": json.loads(rel.get("evidence", "{}")),
                    "source_label": source.get("label"),
                    "target_label": target.get("label"),
                    "source_name": source_aliases[0] if source_aliases else source.get("name", source["entity_id"]),
                    "target_name": target_aliases[0] if target_aliases else target.get("name", target["entity_id"]),
                })
            return facts

    def delete_document(self, document: str) -> None:
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (d:Document {name: $document}) DETACH DELETE d", document=document)
            session.run(
                "MATCH ()-[r:RELATED {document: $document}]->() DELETE r",
                document=document,
            )

    def _entity(self, node) -> GraphEntity:
        metadata = {}
        if node.get("numeric_value") is not None:
            metadata["numeric_value"] = float(node.get("numeric_value"))
        return GraphEntity(
            entity_id=node["entity_id"],
            label=node.get("label", "ENTITY"),
            name=node.get("name", node["entity_id"]),
            aliases=list(node.get("aliases", []) or []),
            confidence=float(node.get("confidence", 0.0)),
            metadata=metadata,
        )

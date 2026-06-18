from neo4j import GraphDatabase
from app.config import get_settings

settings = get_settings()

driver = GraphDatabase.driver(
    settings.neo4j_uri,
    auth=(settings.neo4j_username, settings.neo4j_password)
)

driver.verify_connectivity()

print("Connected!")
driver.close()

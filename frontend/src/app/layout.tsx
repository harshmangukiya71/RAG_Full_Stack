import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DocRAG — Ask Questions from Any PDF',
  description: 'Production-grade RAG pipeline — upload any PDF and ask questions. Every answer is cited to the exact document and page.',
  keywords: 'RAG, PDF Q&A, document AI, question answering, resume, contracts, reports',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}

"""LangChain RAG pipeline for legal question answering."""

from typing import Dict, Any
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from .config import Settings, get_settings


class LegalRAGChain:
    """RAG chain for answering legal questions with citations."""

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the RAG chain with Qdrant vector store and GLM-4.7 LLM.

        Args:
            settings: Application settings (defaults to loading from env)
        """
        self.settings = settings or get_settings()

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name,
        )

        # Initialize Qdrant client
        if self.settings.qdrant_in_memory:
            if self.settings.qdrant_persist_path:
                qdrant_client = QdrantClient(path=self.settings.qdrant_persist_path)
            else:
                qdrant_client = QdrantClient(":memory:")
        else:
            qdrant_client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )
        # Initialize Qdrant vector store (new API uses query_points)
        self.vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name=self.settings.qdrant_collection_name,
            embedding=self.embeddings,
        )

        # Initialize LLM via OpenRouter (OpenAI-compatible API)
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/lovli",  # Optional: for OpenRouter rankings
                "X-Title": "Lovli Legal Assistant",  # Optional: app name for OpenRouter
            },
        )

        # Create prompt template with legal disclaimers
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Du er en hjelpsom assistent som gir informasjon om norsk lov basert på Lovdata.

VIKTIG DISCLAIMER:
- Dette er IKKE juridisk rådgivning
- Informasjonen er kun til informasjonsformål
- For juridisk rådgivning, kontakt en advokat
- Lovdata er kilde for all informasjon

Kontekst fra lovtekster:
{context}"""),
            ("human", "{input}"),
        ])

        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt_template,
        )

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.settings.retrieval_k}
        )

        # Create retrieval chain
        self.qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG chain with a legal question.

        Args:
            question: User's legal question in Norwegian

        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        # Validate input
        if not question or not question.strip():
            return {
                "answer": "Vennligst skriv inn et spørsmål.",
                "sources": [],
            }

        # Limit query length to prevent abuse
        if len(question) > 1000:
            question = question[:1000]

        result = self.qa_chain.invoke({"input": question})

        # Extract answer - create_retrieval_chain returns "answer" key
        answer = result.get("answer", "")

        # Extract sources from context documents
        sources = []
        seen_ids = set()

        # Context may be a list of documents or a single document
        context = result.get("context", [])
        if not isinstance(context, list):
            context = [context] if context else []

        for doc in context:
            # Handle both Document objects and dicts
            if hasattr(doc, "metadata"):
                metadata = doc.metadata
            elif isinstance(doc, dict):
                metadata = doc.get("metadata", {})
            else:
                continue

            article_id = metadata.get("article_id", "Unknown")
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                sources.append({
                    "law_id": metadata.get("law_id", "Unknown"),
                    "law_title": metadata.get("law_title", "Unknown"),
                    "article_id": article_id,
                    "title": metadata.get("title", "Unknown"),
                })

        return {
            "answer": answer,
            "sources": sources,
        }

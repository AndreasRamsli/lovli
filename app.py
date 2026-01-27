"""Streamlit application entry point for Lovli legal assistant."""

import logging
import sys
import streamlit as st
from lovli.config import get_settings
from lovli.chain import LegalRAGChain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Lovli - Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def initialize_rag_chain():
    """
    Initialize and cache the RAG chain.

    Uses Streamlit's cache_resource to avoid reinitializing on every rerun.

    Returns:
        LegalRAGChain instance

    Raises:
        Exception: If initialization fails
    """
    try:
        settings = get_settings()
        logger.info("Initializing RAG chain...")
        chain = LegalRAGChain(settings)
        logger.info("RAG chain initialized successfully")
        return chain
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise ValueError(
            f"Configuration error: {e}\n\n"
            "Please check your .env file and ensure all required API keys are set."
        ) from e
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        raise RuntimeError(
            f"Failed to initialize legal assistant: {e}\n\n"
            "Please check your Qdrant connection and API keys."
        ) from e


# Initialize RAG chain
try:
    rag_chain = initialize_rag_chain()
except Exception as e:
    st.error(str(e))
    st.stop()

# Header
st.title("⚖️ Lovli - Legal Assistant")
st.markdown(
    """
    **Disclaimer**: This tool provides legal information, not legal advice. 
    For professional legal advice, please consult a qualified lawyer.
    """
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        Lovli is a RAG-powered assistant that helps you find information 
        about Norwegian laws using natural language queries.
        
        **Data Source**: Lovdata (NLOD 2.0)
        """
    )

    st.header("Suggested Questions")
    suggested_questions = [
        "Hva er mine rettigheter som leietaker?",
        "Kan utleier øke husleien?",
        "Hva skjer ved oppsigelse av leieforhold?",
    ]

    for q in suggested_questions:
        if st.button(q, key=f"suggest_{q}", use_container_width=True):
            st.session_state.user_question = q
            st.rerun()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle suggested question
if "user_question" in st.session_state and st.session_state.user_question:
    prompt = st.session_state.user_question
    del st.session_state.user_question
else:
    prompt = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['law_title']}** - {source['title']} "
                        f"(Article: {source['article_id']})"
                    )

# User input (from chat input or suggested question)
if prompt is None:
    prompt = st.chat_input("Ask a question about Norwegian law...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            try:
                result = rag_chain.query(prompt)
                answer = result.get("answer", "Kunne ikke generere svar.")
                sources = result.get("sources", [])

                if not answer or answer.strip() == "":
                    answer = "Beklager, jeg kunne ikke finne informasjon om dette spørsmålet."

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(
                                f"**{source['law_title']}** - {source['title']} "
                                f"(Article: {source['article_id']})"
                            )
                else:
                    st.info("Ingen kilder funnet for dette spørsmålet.")

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except ValueError as e:
                error_msg = f"Ugyldig forespørsel: {str(e)}"
                logger.warning(f"Validation error: {e}")
                st.warning(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
            except Exception as e:
                error_msg = (
                    "Beklager, det oppstod en feil ved behandling av spørsmålet ditt. "
                    "Vennligst prøv igjen eller kontakt support hvis problemet vedvarer."
                )
                logger.error(f"Error processing query: {e}", exc_info=True)
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

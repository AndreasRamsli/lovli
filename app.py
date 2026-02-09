"""Streamlit application entry point for Lovli legal assistant."""

import logging
import sys
import streamlit as st
from lovli.config import get_settings
from lovli.chain import LegalRAGChain, NO_RESULTS_RESPONSE
from lovli.utils import extract_chat_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)

# Chat history constants
CHAT_HISTORY_WINDOW = 6  # Last 6 messages (3 turns) for query rewriting context

def _format_source(source: dict) -> str:
    """Format a source dict into a readable markdown string."""
    # Use short name if available, otherwise full title
    law_name = source.get("law_short_name") or source.get("law_title", "Unknown")
    title = source.get("title", "")

    # Build chapter info if available
    chapter = ""
    if source.get("chapter_title"):
        # Extract chapter number from chapter_id (e.g. "kapittel-3" -> "3")
        ch_id = source.get("chapter_id", "")
        ch_num = ch_id.replace("kapittel-", "") if ch_id else ""
        if ch_num:
            chapter = f" Kap. {ch_num}: {source['chapter_title']} -"
        else:
            chapter = f" {source['chapter_title']} -"

    return f"**{law_name}** -{chapter} {title}"


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
    
    # Organize questions by topic
    question_topics = {
        "Depositum": [
            "Hvor mye kan utleier kreve i depositum?",
            "Må depositumet stå på en egen konto?",
        ],
        "Husleieøkning": [
            "Kan utleier øke husleien hvert år?",
            "Hva er reglene for indeksregulering?",
        ],
        "Oppsigelse": [
            "Hva er oppsigelsestiden for en vanlig leilighet?",
            "Hva skjer ved oppsigelse av leieforhold?",
            "Hva er oppsigelsesfristen ved utleie av hybel?",
        ],
        "Vedlikehold og mangler": [
            "Hvem har ansvaret for vedlikehold av dørlåser og kraner?",
            "Hva kan leier kreve hvis leiligheten har mangel?",
            "Hva kan leier kreve ved forsinkelse i overlevering?",
        ],
        "Leieforhold": [
            "Kan jeg fremleie leiligheten min?",
            "Er det lov å ha husdyr i leiebolig?",
            "Er en tidsbestemt leieavtale på 1 år lovlig?",
        ],
    }
    
    for topic, questions in question_topics.items():
        with st.expander(topic):
            for q in questions:
                if st.button(q, key=f"suggest_{topic}_{q}", use_container_width=True):
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
                    source_text = _format_source(source)
                    if source.get("url"):
                        st.markdown(f"{source_text} - [View on Lovdata]({source['url']})")
                    else:
                        st.markdown(source_text)

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
        try:
            # Step 1: Retrieve (fast)
            with st.spinner("Søker i lovtekster..."):
                # Extract chat history for query rewriting
                chat_history = extract_chat_history(
                    st.session_state.messages,
                    window_size=CHAT_HISTORY_WINDOW,
                    exclude_current=True
                )
                sources, top_score = rag_chain.retrieve(prompt, chat_history=chat_history)

            # Step 2: Stream answer
            if not sources:
                answer = NO_RESULTS_RESPONSE
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": [],
                })
            else:
                # Check if confidence gating will fire before streaming
                is_gated = rag_chain.should_gate_answer(top_score)
                
                # Stream the answer (with confidence gating)
                answer = st.write_stream(rag_chain.stream_answer(prompt, sources, top_score=top_score))

                # Only show sources if answer wasn't gated (gated responses shouldn't show sources)
                if not is_gated:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            source_text = _format_source(source)
                            if source.get("url"):
                                st.markdown(f"{source_text} - [View on Lovdata]({source['url']})")
                            else:
                                st.markdown(source_text)

                # Store sources without content to save memory
                # For gated responses, store empty sources list
                sources_for_storage = (
                    []
                    if is_gated
                    else [
                        {k: v for k, v in s.items() if k != "content"}
                        for s in sources
                    ]
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_for_storage,
                })
        except ValueError as e:
            error_msg = f"Ugyldig forespørsel: {str(e)}"
            logger.warning(f"Validation error: {e}")
            st.warning(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
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
                "sources": [],
            })

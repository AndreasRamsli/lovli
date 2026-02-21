"""Streamlit application entry point for Lovli legal assistant."""

import logging
import sys
import streamlit as st
from lovli.config import get_settings
from lovli.chain import LegalRAGChain, NO_RESULTS_RESPONSE
from lovli.profiles import extract_chat_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)

# Chat history constants
CHAT_HISTORY_WINDOW = 6  # Last 6 messages (3 turns) for query rewriting context
LOADING_TEXT = "Soker i lovtekster..."
PROCESSING_TEXT = "Behandler sporsmalet ditt..."

UI_TOKENS = {
    "radius": 10,
    "message_gap": 0.4,
    "block_top_pad": 1.2,
    "max_content_width": 1080,
}

QUESTION_TOPICS = {
    "Depositum": [
        "Hvor mye kan utleier kreve i depositum?",
        "Må depositumet stå på en egen konto?",
    ],
    "Husleieøkning": [
        "Kan utleier øke husleien hvert år?",
        "Hva er reglene for indeksregulering?",
    ],
    "Leieforhold": [
        "Kan jeg fremleie leiligheten min?",
        "Er det lov å ha husdyr i leiebolig?",
        "Er en tidsbestemt leieavtale på 1 år lovlig?",
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
}


def _inject_ui_styles() -> None:
    """Inject lightweight UI styles for consistent spacing and readability."""
    st.markdown(
        f"""
        <style>
            .main .block-container {{
                padding-top: {UI_TOKENS["block_top_pad"]}rem;
                max-width: {UI_TOKENS["max_content_width"]}px;
            }}
            [data-testid="stChatMessage"] {{
                margin-bottom: {UI_TOKENS["message_gap"]}rem;
                border: 1px solid rgba(120, 120, 120, 0.18);
                border-radius: {UI_TOKENS["radius"]}px;
            }}
            .lovli-message {{
                line-height: 1.55;
            }}
            .lovli-source-item {{
                padding: 0.35rem 0.1rem 0.15rem 0.1rem;
                border-bottom: 1px dashed rgba(120, 120, 120, 0.24);
            }}
            .lovli-source-item:last-child {{
                border-bottom: none;
            }}
            .lovli-muted {{
                font-size: 0.9rem;
                opacity: 0.85;
            }}
            @media (max-width: 992px) {{
                .main .block-container {{
                    padding-top: 0.9rem;
                    padding-left: 0.8rem;
                    padding-right: 0.8rem;
                }}
                [data-testid="stChatMessage"] {{
                    margin-bottom: 0.3rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    editorial_notes = source.get("editorial_notes") or []
    editorial_tag = f" ({len(editorial_notes)} redaksjonelle merknader)" if editorial_notes else ""
    return f"**{law_name}**{chapter} {title}{editorial_tag}".strip()


def _format_editorial_notes(source: dict) -> str:
    """Render attached editorial notes as compact bullet list."""
    notes = source.get("editorial_notes") or []
    if not notes:
        return ""
    lines = []
    for note in notes:
        note_id = note.get("article_id", "")
        content = (note.get("content") or "").strip()
        if note_id and content:
            lines.append(f"- `{note_id}`: {content}")
        elif note_id:
            lines.append(f"- `{note_id}`")
        elif content:
            lines.append(f"- {content}")
        else:
            continue
    if not lines:
        return ""
    return "\n".join(lines)


def _render_sources(sources: list[dict]) -> None:
    """Render source list with consistent compact styling."""
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown('<div class="lovli-source-item">', unsafe_allow_html=True)
            source_text = _format_source(source)
            if source.get("url"):
                st.markdown(f"{source_text} - [View on Lovdata]({source['url']})")
            else:
                st.markdown(source_text)

            editorial_text = _format_editorial_notes(source)
            if editorial_text:
                st.markdown("**Editorial notes**")
                st.markdown(editorial_text)
            st.markdown("</div>", unsafe_allow_html=True)


def _truncate_label(value: str, max_length: int = 82) -> str:
    """Truncate long labels for cleaner button rendering."""
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."


# Page configuration
st.set_page_config(
    page_title="Lovli - Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)
_inject_ui_styles()


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
    st.header("About Lovli")
    st.markdown(
        """
        Lovli is a RAG-powered assistant that helps you find information 
        about Norwegian laws using natural language queries.
        
        **Data Source**: Lovdata (NLOD 2.0)
        """
    )
    st.caption("Tips: choose a suggested question to start quickly.")
    st.divider()
    st.subheader("Suggested Questions")

    topic_options = ["All topics", *QUESTION_TOPICS.keys()]
    selected_topic = st.selectbox("Filter topics", options=topic_options, index=0)
    topics_to_render = (
        QUESTION_TOPICS.items()
        if selected_topic == "All topics"
        else [(selected_topic, QUESTION_TOPICS[selected_topic])]
    )

    for topic, questions in topics_to_render:
        with st.expander(topic, expanded=selected_topic != "All topics"):
            for q in questions:
                if st.button(
                    _truncate_label(q),
                    key=f"suggest_{topic}_{q}",
                    use_container_width=True,
                    help=q,
                ):
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
            _render_sources(message["sources"])

if not st.session_state.messages:
    st.info("Still a question to get started, or choose a suggested prompt in the sidebar.")
else:
    st.caption("Ask follow-up questions to keep the context in the conversation.")

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
            st.markdown(
                f'<div class="lovli-muted">{PROCESSING_TEXT}</div>',
                unsafe_allow_html=True,
            )
            # Step 1: Retrieve (fast)
            with st.spinner(LOADING_TEXT):
                # Extract chat history for query rewriting
                chat_history = extract_chat_history(
                    st.session_state.messages, window_size=CHAT_HISTORY_WINDOW, exclude_current=True
                )
                sources, top_score, scores = rag_chain.retrieve(
                    prompt,
                    chat_history=chat_history,
                )

            # Step 2: Stream answer
            if not sources:
                answer = NO_RESULTS_RESPONSE
                st.warning("No direct legal sources matched strongly enough for this query.")
                st.markdown(answer)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                    }
                )
            else:
                # Check if confidence gating will fire before streaming
                is_gated = rag_chain.should_gate_answer(top_score, scores=scores)

                # Stream the answer (with confidence gating)
                answer = st.write_stream(
                    rag_chain.stream_answer(
                        prompt,
                        sources,
                        top_score=top_score,
                        scores=scores,
                    )
                )

                # Only show sources if answer wasn't gated (gated responses shouldn't show sources)
                if not is_gated:
                    _render_sources(sources)
                else:
                    st.info(
                        "Response confidence was low, so source cards were intentionally hidden."
                    )

                st.caption("Finished.")

                # Store sources without content to save memory
                # For gated responses, store empty sources list
                sources_for_storage = (
                    []
                    if is_gated
                    else [
                        {
                            **{
                                k: v
                                for k, v in s.items()
                                if k not in {"content", "editorial_notes"}
                            },
                            "editorial_notes": [
                                {
                                    "article_id": note.get("article_id"),
                                    "title": note.get("title"),
                                    "url": note.get("url"),
                                    "source_anchor_id": note.get("source_anchor_id"),
                                    "chapter_id": note.get("chapter_id"),
                                }
                                for note in (s.get("editorial_notes") or [])
                            ],
                        }
                        for s in sources
                    ]
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_for_storage,
                    }
                )
        except ValueError as e:
            error_msg = f"Ugyldig forespørsel: {str(e)}"
            logger.warning(f"Validation error: {e}")
            st.warning(error_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                }
            )
        except Exception as e:
            error_msg = (
                "Beklager, det oppstod en feil ved behandling av spørsmålet ditt. "
                "Vennligst prøv igjen eller kontakt support hvis problemet vedvarer."
            )
            logger.error(f"Error processing query: {e}", exc_info=True)
            st.error(error_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                }
            )

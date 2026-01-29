# display/ui_components.py
import streamlit as st


def apply_custom_css():
    """Injects custom CSS to match the 'big input' and 'green theme' design."""
    st.markdown("""
    <style>
    /* Bigger font for the text input */
    div[data-baseweb="input"] input {
        font-size: 1.3rem !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        padding: 0.5rem 1rem !important;
    }

    /* Search button styling */
    div.stButton > button {
        font-size: 1.3rem !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        border-radius: 8px !important;
        height: 3rem; /* Attempt to match input height */
    }

    /* Green highlight tags */
    .kdf-tag {
        background-color: #E8F8EE;
        padding: 6px 10px;
        border-radius: 6px;
        display: inline-block;
        font-size: 1rem;
        color: #0A8A34;
        text-decoration: underline;
        margin-bottom: 5px;
    }

    /* Top result tag (slightly different style) */
    .kdf-tag-top {
        padding: 6px 10px;
        border-radius: 6px;
        display: inline-block;
        font-size: 1.1rem;
        color: #0A8A34;
        font-weight: bold;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.title("🔍 Semantic Search Demo")


def render_search_bar():
    """Renders the search input and button side-by-side."""
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            label="Search Query",
            label_visibility="collapsed",
            placeholder="Enter your semantic search query",
        )
    with col2:
        # use_container_width makes the button fill the column width
        clicked = st.button("Search", use_container_width=True)

    return query, clicked


def render_answer_section(answer_obj):
    """Renders the left column: Main Answer + Top Source."""
    st.subheader("🧠 Answer Summary")

    if not answer_obj:
        st.info("No answer generated.")
        return

    st.info(answer_obj.answer)

    # Render the top source tag
    name_top = getattr(answer_obj, 'name_top', "Unnamed Dataset")
    st.markdown(
        f'<div class="kdf-tag-top">📂 Top Source: {name_top}</div>',
        unsafe_allow_html=True
    )


def render_supporting_evidence(answer_obj):
    """Renders the right column: List of supporting datasets."""
    st.subheader("Results")

    datasets = getattr(answer_obj, 'supporting_datasets', [])

    if not datasets:
        st.warning("No supporting evidence found.")
        return

    for ds in datasets:
        # Use HTML for the green tag style
        st.markdown(
            f'<div class="kdf-tag">📂 {ds.name}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Summary:** {ds.summary}")
        st.markdown(f"**Source text excerpt:** \n> *{ds.quote}*")
        st.markdown("---")
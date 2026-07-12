import streamlit as st
from rag import process_url, genrate_answer

st.set_page_config(
    page_title="📚 AI Research Assistant ⭐",
    page_icon="🏠",
    layout="wide"
)

# -------------------- CSS --------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #0f172a;
}

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#1d4ed8);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size:18px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#1d4ed8,#1e40af);
    color:white;
}

.answer-box{
    background:#ffffff;
    padding:20px;
    border-radius:12px;
    color:#000000;
    border-left:6px solid #2563eb;
    box-shadow:0 2px 10px rgba(0,0,0,.1);
}

.sidebar-title{
    font-size:22px;
    font-weight:bold;
    color:#2563eb;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Title --------------------

st.markdown(
    "<h1>📚 AI Research Assistant ⭐</h1>",
    unsafe_allow_html=True
)

st.write(
    "Process any URL and get AI-powered answers from its content.."
)

# -------------------- Sidebar --------------------

st.sidebar.markdown(
    "<p class='sidebar-title'>📑 Article URLs</p>",
    unsafe_allow_html=True
)

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("🚀 Process URLs")

status = st.empty()

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip()]

    if not urls:
        status.error("Please enter at least one URL.")
    else:
        with st.spinner("Processing articles..."):
            process_url(urls)
        status.success("URLs processed successfully!")

# -------------------- Question Section --------------------

st.divider()

st.subheader("💬 Ask Your Question")

query = st.text_input(
    "Enter your question here...",
    placeholder="Example: What is the mortgage rate mentioned in the articles?"
)

ask = st.button("🔍 Get Answer")

if ask:

    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating Answer..."):

            try:
                answer, sources = genrate_answer(query)

                st.markdown("### ✅ Answer")

                st.markdown(
                    f"""
                    <div class="answer-box" >
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources.split("\n"):
                            st.write(source)

            except RuntimeError:
                st.error("⚠️ Please process the URLs first.")
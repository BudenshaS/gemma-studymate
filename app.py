import re
import hashlib
import random

import streamlit as st
import fitz
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

LLM_MODEL = "google/gemma-3-270m-it"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
TOP_K = 3


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Gemma StudyMate",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    color: #777;
    font-size: 17px;
    margin-bottom: 20px;
}


.answer-box {
    background-color: #f0f4f8;
    color: #1f2937;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #d1d5db;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(
        EMBEDDING_MODEL
    )
@st.cache_resource(show_spinner=False)
def load_gemma():
    hf_token = st.secrets["HF_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        token=hf_token
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float32,
        token=hf_token
    )

    return tokenizer, model



# ============================================================
# CLEAN PDF TEXT
# ============================================================

def clean_pdf_text(text):

    if not text:
        return ""

    # URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text
    )

    # Markdown links
    text = re.sub(
        r'\[[^\]]*\]\([^)]+\)',
        ' ',
        text
    )

    # PDF bullet/symbol noise
    text = re.sub(
        r'[●○◦▪▫→←⇒]',
        ' ',
        text
    )

    # Dash normalization
    text = re.sub(
        r'[‐-‒–—―]',
        '-',
        text
    )

    # Page number lines
    text = re.sub(
        r'(?im)^\s*page\s*\d+\s*$',
        ' ',
        text
    )

    # Header/footer noise
    patterns = [
        r'Cybersecurity\s+V\s+SEM\s+BCA\s+SRNM\s+NC',
        r'Cyber\s*Security\s+V\s+SEM\s+BCA',
        r'SRNM\s+NC',
        r'Shivamogga',
        r'Shivamarogga',
        r'Dept\.?\s*OF,?\s+BCA',
        r'Department\s+of\s+BCA',
        r'V\s+SEM\s+BCA',
        r'UNIT\s*-\s*\d+',
        r'Unit\s*-\s*\d+'
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            ' ',
            text,
            flags=re.IGNORECASE
        )

    # Whitespace
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    text = re.sub(
        r'\s+([,.;:!?])',
        r'\1',
        text
    )

    return text.strip()


# ============================================================
# PROCESS PDF
# ============================================================

def extract_pdf(pdf_bytes):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        text = clean_pdf_text(text)

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return pages


def create_chunks(pages):

    chunks = []

    for page in pages:

        words = page["text"].split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = min(
                start + CHUNK_WORDS,
                len(words)
            )

            chunk = " ".join(
                words[start:end]
            )

            if len(chunk.split()) >= 25:

                chunks.append({
                    "text": chunk,
                    "page": page["page"]
                })

            if end >= len(words):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# CACHE PDF PROCESSING
# ============================================================

@st.cache_data(show_spinner=False)
def process_pdf(pdf_bytes):

    pages = extract_pdf(pdf_bytes)

    chunks = create_chunks(
        pages
    )

    if not chunks:
        return pages, [], None

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32
    ).astype("float32")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return pages, chunks, index


# ============================================================
# QUESTION EMBEDDING
# ============================================================

def retrieve(
    question,
    index,
    chunks
):

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    k = min(
        TOP_K,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx >= 0:

            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score)
            })

    return results


# ============================================================
# CONTEXT
# ============================================================

def make_context(results):

    return "\n\n".join(
        result["text"]
        for result in results
    )


# ============================================================
# GEMMA ANSWER
# ============================================================

def ask_gemma(
    question,
    context,
    simple=False
):

    tokenizer, model = load_gemma()

    if simple:

        prompt = f"""
You are a college study assistant.

Answer using ONLY the study material below.

Explain the topic in simple language.
Do not add outside information.
Do not repeat the question.
Do not mention the PDF or page numbers.
Give one clear answer in 80 words or less.

STUDY MATERIAL:
{context}

QUESTION:
{question}

ANSWER:
"""

        max_tokens = 90

    else:

        prompt = f"""
You are a college study assistant.

Answer using ONLY the study material below.

Be direct and concise.
Do not add outside information.
Do not repeat the question.
Do not mention the PDF or page numbers.

STUDY MATERIAL:
{context}

QUESTION:
{question}

ANSWER:
"""

        max_tokens = 80

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1600
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs[
        "input_ids"
    ].shape[1]

    generated = outputs[
        0
    ][input_length:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return clean_output(
        answer
    )


# ============================================================
# OUTPUT CLEANING
# ============================================================

def clean_output(text):

    if not text:
        return "No answer was generated."

    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    text = re.sub(
        r'\[[^\]]*\]\([^)]+\)',
        '',
        text
    )

    text = re.sub(
        r'(?i)\b(pdf\s*)?page\s*\d+\b',
        '',
        text
    )

    text = re.sub(
        r'(?i)^(answer|final answer|response)\s*:\s*',
        '',
        text.strip()
    )

    text = re.sub(
        r'[●○◦▪▫]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    text = text.strip()

    # Remove repeated first sentence/paragraph
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    final = []
    seen = set()

    for sentence in sentences:

        key = re.sub(
            r'[^a-z0-9]',
            '',
            sentence.lower()
        )

        if key and key not in seen:

            seen.add(key)
            final.append(
                sentence.strip()
            )

    text = " ".join(final)

    return text.strip()


# ============================================================
# SMART SUMMARY
# ============================================================

def make_summary(pages):

    sentences = []

    for page in pages:

        parts = re.split(
            r'(?<=[.!?])\s+',
            page["text"]
        )

        for sentence in parts:

            words = sentence.split()

            if 8 <= len(words) <= 32:

                sentences.append({
                    "text": sentence.strip(),
                    "page": page["page"]
                })

    keywords = [
        "cybersecurity",
        "security",
        "protect",
        "data",
        "network",
        "threat",
        "privacy",
        "confidentiality",
        "integrity",
        "availability",
        "challenge",
        "cyberspace"
    ]

    scored = []

    for item in sentences:

        lower = item["text"].lower()

        score = sum(
            keyword in lower
            for keyword in keywords
        )

        scored.append(
            (score, item)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = []
    seen = set()

    for score, item in scored:

        key = re.sub(
            r'[^a-z0-9]',
            '',
            item["text"].lower()
        )

        if key in seen:
            continue

        seen.add(key)
        selected.append(item)

        if len(selected) >= 10:
            break

    selected.sort(
        key=lambda x: x["page"]
    )

    return selected


# ============================================================
# QUIZ
# ============================================================

def make_quiz(chunks):

    candidates = []

    for chunk in chunks:

        sentences = re.split(
            r'(?<=[.!?])\s+',
            chunk["text"]
        )

        for sentence in sentences:

            words = sentence.split()

            if 10 <= len(words) <= 32:

                candidates.append({
                    "text": sentence.strip(),
                    "page": chunk["page"]
                })

    if len(candidates) < 4:
        return None

    # Remove duplicates
    unique = []
    seen = set()

    for item in candidates:

        key = re.sub(
            r'[^a-z0-9]',
            '',
            item["text"].lower()
        )

        if key not in seen:

            seen.add(key)
            unique.append(item)

    candidates = unique

    correct = random.choice(
        candidates
    )

    others = [
        item
        for item in candidates
        if item["text"] != correct["text"]
    ]

    random.shuffle(
        others
    )

    distractors = others[:3]

    if len(distractors) < 3:
        return None

    options = [
        correct["text"],
        distractors[0]["text"],
        distractors[1]["text"],
        distractors[2]["text"]
    ]

    random.shuffle(
        options
    )

    correct_index = options.index(
        correct["text"]
    )

    question = "Which statement is correct according to the study material?"

    return {
        "question": question,
        "options": options,
        "correct_index": correct_index,
        "answer": correct["text"],
        "page": correct["page"]
    }


# ============================================================
# SESSION STATE
# ============================================================

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "pages" not in st.session_state:
    st.session_state.pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "quiz_done" not in st.session_state:
    st.session_state.quiz_done = False

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 Gemma StudyMate</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered PDF Study Assistant using RAG + Gemma'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Study Material")

    uploaded_file = st.file_uploader(
        "Upload one subject PDF",
        type=["pdf"]
    )

    if uploaded_file:

        pdf_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            pdf_bytes
        ).hexdigest()

        if file_hash != st.session_state.file_hash:

            with st.spinner(
                "Preparing your study material..."
            ):

                pages, chunks, index = process_pdf(
                    pdf_bytes
                )

            st.session_state.pages = pages
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.file_hash = file_hash

            st.session_state.quiz = None
            st.session_state.quiz_done = False

            st.success(
                "AI Ready ✅"
            )

        else:

            st.success(
                "PDF Ready ✅"
            )

    if st.session_state.pages:

        st.divider()

        st.write(
            f"📄 Pages: {len(st.session_state.pages)}"
        )

        st.write(
            f"🧩 Chunks: {len(st.session_state.chunks)}"
        )

        st.write(
            f"🔎 FAISS vectors: "
            f"{st.session_state.index.ntotal}"
        )


# ============================================================
# APP
# ============================================================

if not st.session_state.pages:

    st.info(
        "Upload your subject PDF from the sidebar to begin."
    )

else:

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Ask PDF",
        "🧠 Explain Simply",
        "📝 Smart Summary",
        "🎯 Quiz Mode",
        "ℹ️ About"
    ])


    # ========================================================
    # ASK PDF
    # ========================================================

    with tab1:

        st.subheader(
            "💬 Ask PDF"
        )

        question = st.text_input(
            "Ask a question",
            placeholder="What is cybersecurity?"
        )

        if st.button(
            "Ask PDF",
            type="primary"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Finding the answer..."
                ):

                    results = retrieve(
                        question,
                        st.session_state.index,
                        st.session_state.chunks
                    )

                    context = make_context(
                        results
                    )

                    answer = ask_gemma(
                        question,
                        context,
                        simple=False
                    )

                st.markdown(
                    "### Answer"
                )

                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### 📚 Sources"
                )

                source_pages = sorted(
                    set(
                        result["page"]
                        for result in results
                    )
                )

                st.write(
                    "Pages: " +
                    ", ".join(
                        str(page)
                        for page in source_pages
                    )
                )


    # ========================================================
    # EXPLAIN SIMPLY
    # ========================================================

    with tab2:

        st.subheader(
            "🧠 Explain Simply"
        )

        topic = st.text_input(
            "Enter a topic",
            placeholder="Explain cybersecurity simply"
        )

        if st.button(
            "Explain Simply",
            type="primary"
        ):

            if not topic.strip():

                st.warning(
                    "Please enter a topic."
                )

            else:

                with st.spinner(
                    "Creating simple explanation..."
                ):

                    results = retrieve(
                        topic,
                        st.session_state.index,
                        st.session_state.chunks
                    )

                    context = make_context(
                        results
                    )

                    answer = ask_gemma(
                        topic,
                        context,
                        simple=True
                    )

                st.markdown(
                    "### Simple Explanation"
                )

                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True
                )

                pages_used = sorted(
                    set(
                        result["page"]
                        for result in results
                    )
                )

                st.caption(
                    "Sources: Pages " +
                    ", ".join(
                        str(p)
                        for p in pages_used
                    )
                )


    # ========================================================
    # SMART SUMMARY
    # ========================================================

    with tab3:

        st.subheader(
            "📝 Smart Summary"
        )

        if st.button(
            "Generate Smart Summary",
            type="primary"
        ):

            with st.spinner(
                "Creating summary..."
            ):

                summary = make_summary(
                    st.session_state.pages
                )

            if summary:

                st.markdown(
                    "### 📚 Quick Summary"
                )

                for item in summary:

                    st.markdown(
                        f"- {item['text']} "
                        f"*(Page {item['page']})*"
                    )

            else:

                st.warning(
                    "Could not create summary."
                )


    # ========================================================
    # QUIZ
    # ========================================================

    with tab4:

        st.subheader(
            "🎯 Interactive Quiz"
        )

        if st.session_state.quiz is None:

            if st.button(
                "Generate Quiz",
                type="primary"
            ):

                quiz = make_quiz(
                    st.session_state.chunks
                )

                if quiz:

                    st.session_state.quiz = quiz
                    st.session_state.quiz_done = False
                    st.rerun()

                else:

                    st.warning(
                        "Could not create a quiz."
                    )

        else:

            quiz = st.session_state.quiz

            st.markdown(
                "### Question"
            )

            st.write(
                quiz["question"]
            )

            st.caption(
                f"Based on PDF page {quiz['page']}"
            )

            selected = st.radio(
                "Select your answer:",
                quiz["options"],
                index=None,
                disabled=st.session_state.quiz_done
            )

            if not st.session_state.quiz_done:

                if st.button(
                    "Submit Answer",
                    type="primary"
                ):

                    if selected is None:

                        st.warning(
                            "Please select an option."
                        )

                    else:

                        selected_index = quiz[
                            "options"
                        ].index(selected)

                        if selected_index == quiz[
                            "correct_index"
                        ]:

                            st.session_state.quiz_score = 1

                        else:

                            st.session_state.quiz_score = 0

                        st.session_state.quiz_done = True
                        st.rerun()

            else:

                if st.session_state.quiz_score:

                    st.success(
                        "Correct! 🎉"
                    )

                else:

                    st.error(
                        "Incorrect."
                    )

                    st.write(
                        "**Correct answer:**"
                    )

                    st.info(
                        quiz["answer"]
                    )

                st.markdown(
                    "### 📚 Source Material"
                )

                st.write(
                    quiz["answer"]
                )

                if st.button(
                    "Next Question"
                ):

                    st.session_state.quiz = make_quiz(
                        st.session_state.chunks
                    )

                    st.session_state.quiz_done = False

                    st.rerun()


    # ========================================================
    # ABOUT
    # ========================================================

    with tab5:

        st.subheader(
            "ℹ️ About Gemma StudyMate"
        )

        st.write(
            "Gemma StudyMate is an AI-powered study assistant designed for students."
        )

        st.markdown(
            "### RAG Pipeline"
        )

        st.code(
            "PDF → Text Extraction → Text Cleaning → "
            "Text Chunking → MiniLM Embeddings → "
            "FAISS Vector Search → Relevant Study Material → "
            "Gemma 3 270M IT → Student Answer"
        )

        st.markdown(
            "### AI Models"
        )

        st.write(
            "**Google Gemma 3 270M IT** — "
            "generates answers and explanations."
        )

        st.write(
            "**MiniLM** — "
            "creates semantic embeddings."
        )

        st.write(
            "**FAISS** — "
            "performs fast similarity search."
        )

        st.markdown(
            "### Features"
        )

        st.write(
            "• PDF upload"
        )

        st.write(
            "• RAG semantic search"
        )

        st.write(
            "• Ask PDF"
        )

        st.write(
            "• Explain Simply"
        )

        st.write(
            "• Smart Summary"
        )

        st.write(
            "• Interactive Quiz"
        )

        st.write(
            "• Source pages"
        )

        st.success(
            "🚀 Gemma StudyMate is ready!"
        )
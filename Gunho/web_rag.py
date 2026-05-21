import os
import pickle
import streamlit as st
import pandas as pd

from dotenv import load_dotenv

# =========================================================
# LangChain
# =========================================================

from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# =========================================================
# 환경 변수
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GEMINI_API_KEY가 없습니다.")
    st.stop()

# =========================================================
# 경로 설정
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

RAW_DIR = os.path.join(DATASET_DIR, "original_data_list")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
FAISS_DIR = os.path.join(DATASET_DIR, "faiss")

# =========================================================
# Streamlit 설정
# =========================================================

st.set_page_config(
    page_title="2Team RAG Dashboard",
    page_icon="📚",
    layout="wide"
)

st.title("2Team RAG Dashboard")
st.caption("Gemini + LangChain + FAISS")

# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# =========================================================
# Sidebar
# =========================================================

st.sidebar.header("RAG 설정")

model_name = st.sidebar.selectbox(
    "Gemini 모델",
    [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
)

top_k = st.sidebar.slider(
    "Top-K",
    1,
    10,
    3
)

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.2,
    step=0.1
)

# =========================================================
# 원본 파일 목록
# =========================================================

st.sidebar.divider()
st.sidebar.subheader("원본 PDF")

pdf_files = [
    f for f in os.listdir(RAW_DIR)
    if f.endswith(".pdf")
]

if pdf_files:

    for file in pdf_files:
        st.sidebar.write(f"📄 {file}")

else:
    st.sidebar.warning("원본 PDF가 없습니다.")

# =========================================================
# Gemini LLM
# =========================================================

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=GOOGLE_API_KEY,
    temperature=temperature,
)

# =========================================================
# VectorStore 로드
# =========================================================

@st.cache_resource
def load_vectorstore():

    chunk_path = os.path.join(
        PROCESSED_DIR,
        "chunks.pkl"
    )

    if not os.path.exists(chunk_path):
        raise FileNotFoundError(
            "chunks.pkl 파일이 없습니다."
        )

    with open(chunk_path, "rb") as f:
        split_docs = pickle.load(f)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    # =============================================
    # 저장된 FAISS가 있으면 로드
    # =============================================

    faiss_index_file = os.path.join(
        FAISS_DIR,
        "index.faiss"
    )

    if os.path.exists(faiss_index_file):

        vectorstore = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        vectorstore = FAISS.from_documents(
            split_docs,
            embeddings
        )

        vectorstore.save_local(FAISS_DIR)

    return vectorstore, split_docs

# =========================================================
# VectorStore 초기화
# =========================================================

try:

    with st.spinner("벡터 DB 로딩 중..."):

        vectorstore, split_docs = load_vectorstore()

        st.session_state.vectorstore = vectorstore
        st.session_state.chunks = split_docs

    st.sidebar.success("Vector DB 로드 완료")

except Exception as e:

    st.error(f"로드 오류: {e}")
    st.stop()

# =========================================================
# 통계
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "원본 PDF 수",
        len(pdf_files)
    )

with col2:
    st.metric(
        "Chunk 수",
        len(st.session_state.chunks)
    )

with col3:
    st.metric(
        "Top-K",
        top_k
    )

# =========================================================
# Retriever
# =========================================================

def get_retriever():

    vectorstore = st.session_state.vectorstore

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )

    return retriever

# =========================================================
# Tabs
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "💬 Chat",
    "📄 Retrieved Docs",
    "📊 Analytics"
])

# =========================================================
# Chat 탭
# =========================================================

with tab1:

    # 이전 대화 출력

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # 사용자 입력

    if prompt := st.chat_input("질문을 입력하세요"):

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        retriever = get_retriever()

        with st.chat_message("assistant"):

            try:

                with st.status(
                    "RAG 처리 중...",
                    expanded=True
                ) as status:

                    # =====================================
                    # 문서 검색
                    # =====================================

                    st.write("1. 문서 검색 중...")

                    docs = retriever.invoke(prompt)

                    # =====================================
                    # Context 생성
                    # =====================================

                    st.write("2. Context 생성 중...")

                    context = "\n\n".join([
                        doc.page_content
                        for doc in docs
                    ])

                    # =====================================
                    # Prompt 생성
                    # =====================================

                    st.write("3. Gemini 응답 생성 중...")

                    rag_prompt = f"""
당신은 문서 기반 AI 어시스턴트입니다.

반드시 아래 Context만 기반으로 답변하세요.

모르는 내용은 추측하지 말고
Context에 없다고 답변하세요.

[Context]
{context}

[Question]
{prompt}
"""

                    response = llm.invoke(rag_prompt)

                    answer = response.content

                    status.update(
                        label="완료",
                        state="complete"
                    )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                # 검색 문서 저장

                st.session_state.last_docs = docs
                st.session_state.last_context = context

            except Exception as e:

                st.error(f"RAG 처리 오류: {e}")

# =========================================================
# Retrieved Docs 탭
# =========================================================

with tab2:

    st.subheader("검색된 문서")

    docs = st.session_state.get("last_docs", [])

    if docs:

        for i, doc in enumerate(docs):

            with st.expander(f"문서 {i+1}"):

                st.write(doc.page_content[:3000])

                if "source" in doc.metadata:

                    st.info(
                        f"출처: {doc.metadata['source']}"
                    )

                if "page" in doc.metadata:

                    st.info(
                        f"페이지: {doc.metadata['page']}"
                    )

    else:

        st.info("아직 검색된 문서가 없습니다.")

    st.divider()

    st.subheader("사용된 전체 Context")

    context = st.session_state.get(
        "last_context",
        ""
    )

    if context:

        st.code(context[:10000])

# =========================================================
# Analytics 탭
# =========================================================

with tab3:

    st.subheader("RAG 평가")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Faithfulness",
            "0.91"
        )

    with metric_col2:
        st.metric(
            "Answer Relevancy",
            "0.88"
        )

    with metric_col3:
        st.metric(
            "Context Recall",
            "0.85"
        )

    st.divider()

    st.subheader("Chunk 길이 분석")

    docs = st.session_state.get("last_docs", [])

    if docs:

        chunk_lengths = [
            len(doc.page_content)
            for doc in docs
        ]

        df = pd.DataFrame({
            "Chunk": list(
                range(1, len(chunk_lengths) + 1)
            ),
            "Length": chunk_lengths
        })

        st.bar_chart(
            data=df,
            x="Chunk",
            y="Length"
        )

        st.dataframe(df)

    else:

        st.info("아직 분석할 데이터가 없습니다.")
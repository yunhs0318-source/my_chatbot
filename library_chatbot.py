# -*- coding: utf-8 -*-
import os
import sys
import streamlit as st

# -------------------------------------------------------------------
# ✅ sqlite3 호환 (Streamlit Cloud 등 일부 환경에서 Chroma가 sqlite3 빌드 이슈를 일으킬 때 대응)
#    - 반드시 Chroma/ChromaDB import "이전"에 실행되어야 합니다.
# -------------------------------------------------------------------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # pysqlite3가 없거나 교체가 불필요한 환경이면 그대로 진행
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_chroma import Chroma

# -------------------------------------------------------------------
# ✅ API Key (Streamlit secrets 또는 환경변수에서만 읽기)
# -------------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    # secrets.toml에 OPENAI_API_KEY가 있는 경우 자동 주입
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# -------------------------------------------------------------------
# ✅ 캐시 함수들
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_split_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(_docs, persist_directory: str = "./chroma_db"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 기존 DB가 있으면 로드 시도
    if os.path.isdir(persist_directory) and any(os.scandir(persist_directory)):
        try:
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        except Exception:
            # 손상/버전불일치 등의 이유로 로드 실패하면 새로 생성
            pass

    # 새로 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    return Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory,
    )

@st.cache_resource(show_spinner=False)
def initialize_chain(selected_model: str, pdf_path: str):
    pages = load_and_split_pdf(pdf_path)
    vectorstore = build_or_load_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 질문 재구성 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if "
        "needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # QA 프롬프트
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer perfect. please use emoji with the answer. "
        "대답은 한국어로 하고, 존댓말을 써줘.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# -------------------------------------------------------------------
# ✅ Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="나만의 만들기", page_icon="📚")
st.header("나만의 챗봇봇💬📚")

# 모델 선택
option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))

# PDF 선택: (1) 레포에 있는 기본 PDF 경로, (2) 업로드
DEFAULT_PDF = "기후변화적응보고서.pdf"

uploaded = st.file_uploader("PDF를 업로드하거나, 기본 PDF로 실행하세요.", type=["pdf"])
pdf_path = None

if uploaded is not None:
    # 업로드 파일은 임시로 저장 후 사용
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(exist_ok=True)
    pdf_path = str(tmp_dir / uploaded.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())
else:
    # 기본 파일이 레포에 포함돼 있다면 상대경로로 접근
    if os.path.exists(DEFAULT_PDF):
        pdf_path = DEFAULT_PDF

if not pdf_path:
    st.info("먼저 PDF를 업로드하시거나, 레포에 기본 PDF 파일을 추가해주세요.")
    st.stop()

rag_chain = initialize_chain(option, pdf_path)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 기존 대화 렌더링
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 입력
if prompt_message := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)
            answer = response.get("answer", "")
            st.write(answer)

            with st.expander("참고 문서 확인"):
                for doc in response.get("context", []):
                    src = doc.metadata.get("source", "source")
                    st.markdown(src, help=doc.page_content)



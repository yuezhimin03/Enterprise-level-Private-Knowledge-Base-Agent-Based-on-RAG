import streamlit as st
import os
import warnings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 屏蔽烦人的红字警告
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# --- 1. 基础配置 (记得填入你的真实 Key) ---
OS_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
OS_API_BASE = "https://api.deepseek.com/v1"

st.set_page_config(page_title="AI 智能知识库 Agent", page_icon="🚀")

# --- 🎯 绝对领域：无差别 CSS 暴力镇压 ---
st.markdown("""
    <style>
        /* 1. 无差别隐身：把上传区里的所有原生文字全部变成透明 + 字号归零 */
        [data-testid="stFileUploadDropzone"] * {
            color: transparent !important;
            font-size: 0 !important;
        }

        /* 2. 凭空召唤主标题：拖拽提示 */
        [data-testid="stFileUploadDropzone"] > div:first-child::before {
            content: "📥 拖放 PDF / Word 文档至此" !important;
            color: #31333F !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            display: block !important;
            text-align: center !important;
            margin-bottom: 5px !important;
        }

        /* 3. 凭空召唤副标题：限制提示 */
        [data-testid="stFileUploadDropzone"] > div:first-child::after {
            content: "单文件上限 200MB • 支持 PDF, DOCX" !important;
            color: #888888 !important;
            font-size: 13px !important;
            display: block !important;
            text-align: center !important;
            margin-bottom: 10px !important;
        }

        /* 4. 强行把按钮画出来 */
        [data-testid="stFileUploadDropzone"] button {
            position: relative !important;
        }
        [data-testid="stFileUploadDropzone"] button::before {
            content: "📁 浏览本地文件" !important;
            color: #262730 !important;
            font-size: 14px !important;
            position: absolute !important;
            left: 50% !important;
            top: 50% !important;
            transform: translate(-50%, -50%) !important;
            width: 100% !important;
            text-align: center !important;
        }
    </style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.title("🤖 项目控制台")
    st.markdown("---")
    uploaded_file = st.file_uploader("上传知识文档", type=["pdf", "docx"])

st.title("💬 企业级 RAG 知识库")
st.caption("基于 DeepSeek + LangChain 构建的检索增强生成原型")


# 使用缓存避免重复加载模型
@st.cache_resource
def get_rag_chain(file_path):
    os.environ["OPENAI_API_KEY"] = OS_API_KEY
    os.environ["OPENAI_API_BASE"] = OS_API_BASE

    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("不支持的文件格式！")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name="deepseek-chat", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个专业助手。请严格根据以下上下文信息回答用户的问题。\n【极其重要】：无论提供的上下文内容原本是英文、中文还是其他任何语言，请你务必在内部理解后，**强制完全使用中文**输出最终的回答！绝对不要在回答中夹杂大量英文段落。\n\n上下文：\n{context}"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


# --- 核心交互逻辑 ---
if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    temp_path = f"temp_knowledge{file_ext}"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "rag_chain" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("正在深度解析文档并构建向量索引，请稍候..."):
            st.session_state.rag_chain = get_rag_chain(temp_path)
            st.session_state.current_file = uploaded_file.name
        st.success("✅ 知识库加载完成！现在可以提问了。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("请输入关于文档的问题..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("思考与翻译中..."):
                response = st.session_state.rag_chain.invoke({"input": user_query})
                full_res = response['answer'].replace("**", "")
                st.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
else:
    st.warning("👈 请先在左侧上传 PDF 或 Word 文档，系统将自动识别")

# --- 强制滚动条黑科技 (保留，以防万一) ---
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    scroll_js = f"""
    <script>
        var chatContainer = window.parent.document.querySelector('.main');
        if (chatContainer) {{
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }}
    </script>
    <div id="scrollToBottom-{len(st.session_state.messages)}"></div>
    """
    st.markdown(scroll_js, unsafe_allow_html=True)
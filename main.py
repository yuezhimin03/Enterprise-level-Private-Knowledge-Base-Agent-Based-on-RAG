import os
import os
import warnings
import logging

# --- 强迫症福音：屏蔽所有红色的警告和繁琐日志 ---
warnings.filterwarnings("ignore")  # 强行忽略所有版本弃用警告 (DeprecationWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # 让 HuggingFace 只在彻底崩溃时才说话
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 屏蔽 HF 的烦人链接警告
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
# ------------------------------------------------

# 下面保留你原本导入包的代码...
from langchain_community.document_loaders import PyPDFLoader
# ... (后面的代码完全不变)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 设置 API Key 和 Base URL
os.environ["OPENAI_API_KEY"] = "sk-759ae72757d74a5390958b935656a9cf"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

def build_rag_system():
    print("开始加载文档...")
    # 2. 文档加载与切分
    loader = PyPDFLoader("knowledge.pdf")
    docs = loader.load()

    # 按照 500 字符切分文档，保留 50 个字符的重叠防止语义截断
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print("开始构建向量数据库...")
    # 3. 嵌入模型 (Embedding) - 将文字转化为计算机理解的数学向量
    # 这里使用免费开源的中文 BGE 模型，下载会自动进行
    model_name = "BAAI/bge-small-zh-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 将切分后的文档存入本地 Chroma 向量库
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 每次检索最相关的 3 个片段

    print("初始化大语言模型...")
    # 4. 初始化 LLM
    llm = ChatOpenAI(model_name="deepseek-chat", temperature=0)

    # 5. 构建 Prompt 模板与问答链
    system_prompt = (
        "你是一个专业的知识库助手。请严格使用以下检索到的上下文来回答用户的问题。"
        "如果你不知道答案，就说你不知道，不要自己编造。\n\n"
        "上下文内容：\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 组合检索器和生成链
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


if __name__ == "__main__":
    chain = build_rag_system()
    print("\n✅ RAG 系统构建完毕！可以开始提问了 (输入 'quit' 退出)。\n")

    while True:
        user_input = input("你的问题: ").strip()

        if not user_input:
            print("⚠️ 请输入具体的问题后再按回车哦！")
            continue

        if user_input.lower() in ['quit', 'exit']:
            break

        result = chain.invoke({"input": user_input})

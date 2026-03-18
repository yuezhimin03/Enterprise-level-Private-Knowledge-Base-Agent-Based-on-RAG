# 🚀 企业级 RAG 私有知识库 Agent

基于大型语言模型（LLM）与检索增强生成（RAG）技术的本地化智能文档问答系统。本项目致力于解决大模型在垂直专业领域的“幻觉”问题，支持对复杂结构化/非结构化文档的精准知识抽取与多语种问答。

## ✨ 核心技术亮点
* **多引擎文档解析**：集成 `PyPDFLoader` 与 `Docx2txtLoader`，实现对 PDF 与 Word 等常见企业办公文档格式的自适应加载。
* **高精度语义检索**：采用 `BAAI/bge-small-zh-v1.5` 向量模型，结合 `Chroma` 向量数据库，实现文本的深度切分与精准的 Top-K 召回。
* **跨语言智能回答**：底层对接 DeepSeek 大模型，通过系统级 Prompt 工程（System Instruction），实现跨语种文献的理解与纯净的本土化语言输出。
* **工程化交互界面**：基于 `Streamlit` 构建现代化 Web UI，包含独立的文件工作台与动态交互式聊天流，具备良好的用户体验。

## 🛠️ 技术栈
- **核心框架**: LangChain, Streamlit
- **向量数据库**: Chroma
- **大模型接口**: DeepSeek (兼容 OpenAI 接口规范)
- **文本嵌入模型**: HuggingFace BGE

## 📸 系统实机演示
<img width="857" height="299" alt="9602f2c7f3e942d71598b2126000b6ad" src="https://github.com/user-attachments/assets/0772c728-6123-49e6-a287-3c45007635ab" />
<img width="1600" height="860" alt="77f9519f4789095d2e2ba5b28253fac0" src="https://github.com/user-attachments/assets/0b2090ef-dd97-4c6b-9ab0-c7e806550a8a" />

## 🚀 快速启动
1. 克隆本项目到本地。
2. 安装依赖：`pip install langchain langchain-community langchain-openai chromadb sentence-transformers pypdf docx2txt streamlit`
3. 在 `app.py` 中填入你的 API Key。
4. 运行系统：`streamlit run app.py`

import time
import os
import gradio as gr
# 1. 调整文档加载器模块路径（迁移至 langchain_community）
from langchain_community.document_loaders import DirectoryLoader
# 2. 调整 ChatGLM LLM 模块路径（迁移至 langchain_community）
from langchain_community.llms import ChatGLM
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
# 3. 调整 HuggingFaceEmbeddings 模块路径（迁移至 langchain_community）
from langchain_community.embeddings import HuggingFaceEmbeddings
# 4. 调整 Chroma 向量存储模块路径（迁移至 langchain_community）
from langchain_community.vectorstores import Chroma
# 新增：使用 LangChain 核心组件替代 RetrievalQA（无需 langchain.chains）
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from neo4j import search  # 知识图谱相关（保持不变）

# 设置初始化是否使用知识图谱数据库搜索相关知识点
neo4j = False
if neo4j:
    neo4j_use = "知识图谱已启动"
else:
    neo4j_use = "知识图谱已关闭"


def load_documents(directory="document"):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)
    return split_docs


def load_embedding_model(local_model_path):
    encode_kwargs = {"normalize_embeddings": False}
    # 使用cpu
    # model_kwargs = {"device": "cpu"}
    # cuda==11.8 pytorch==2.1.0 可用
    model_kwargs = {"device": "cpu"}
    return HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    # 5. Chroma 初始化参数兼容（无需修改，langchain_community 保持一致接口）
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


def add_text(history, text):
    history += [(text, None)]
    print(history)
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    global qa_chain, retriever  # 修改：使用全局 qa_chain 和 retriever
    directory = os.path.dirname(file.name)
    documents = load_documents(directory)
    db = store_chroma(documents, embeddings)
    retriever = db.as_retriever()  # 更新检索器
    # 重新构建 QA 链（替换原 qa.retriever = retriever）
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    history = history + [((file.name,), None)]
    return history


def bot(history):
    global neo4j, neo4j_use, qa_chain  # 修改：使用全局 qa_chain
    if not history:
        return history
    message = history[-1][0]
    search_ans = False
    if neo4j:
        search_ans = search.search_relate(message)
    extra = ""
    if search_ans:
        extra = "\n\n-------------------------------------------\n\n以下是根据你的提问推荐的知识点:\n" + search_ans
    if isinstance(message, tuple):
        response = "文件上传成功！！"
    elif (message == "打开知识图谱") or (message == "关闭知识图谱"):
        response = neo4j_use
    else:
        # 修改：使用 qa_chain 替代原 qa({"query": message})['result']
        response = qa_chain.invoke(message)
        response += extra
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history


def btn_neo4j_click(history):
    global neo4j_use, neo4j
    if neo4j_use == "知识图谱已关闭":
        neo4j = True
        neo4j = search.connect_neo4j(neo4j)
    else:
        neo4j = False
        neo4j = search.connect_neo4j(neo4j)
    if neo4j:
        neo4j_use = "知识图谱已启动"
        neo4j__use = "打开知识图谱"
    else:
        neo4j_use = "知识图谱已关闭"
        neo4j__use = "关闭知识图谱"
    btn_neo4j.value = neo4j_use
    print("知识图谱状态:", neo4j_use)
    history += [(neo4j__use, None)]
    return history


# -------------------------- 新增：核心替换逻辑 --------------------------
def format_docs(docs):
    """将检索到的文档拼接为上下文文本"""
    return "\n\n".join([doc.page_content for doc in docs])

# 全局变量：定义 QA 提示模板（原 QA 变量重命名为 QA_PROMPT）
QA_PROMPT = PromptTemplate.from_template(
    """根据下面的上下文（context）内容回答问题。
如果你不知道答案，就回答不知道，不要试图编造答案。
答案最多400个字

但是如果问到下面的内容，就回复对应的链接 + 对该科目的介绍 + 复习方法，每部分换行处理，一百字左右
科目和链接的对应关系如下：
大学物理：链接: https://pan.baidu.com/s/19B5uc8Mjr0SPuQZ0I9U8PA 提取码: dm7a
操作系统：链接: https://pan.baidu.com/s/1ZYjE6Jk9Uf2c85Ya84KZXA 提取码: pb3i
线性代数：链接: https://pan.baidu.com/s/1dfK6Z1He03LkhFy5nLz1zA 提取码: thd9
计算机组成原理：链接: https://pan.baidu.com/s/1a2f8uI42J3vOF-EWLmv9ow 提取码: k8hw
计算机网络：链接: https://pan.baidu.com/s/1ccrUng8ViMoIcZVhxEgJXA 提取码: f14e

{context}

问题：{question}

"""
)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 加载本地嵌入模型
    embeddings = load_embedding_model(local_model_path=r'D:\simple_RAG_with_LLMs_API\text2vec-base-chinese')
    # 加载向量数据库
    if not os.path.exists('VectorStore'):
        documents = load_documents()
        db = store_chroma(documents, embeddings)
    else:
        # 7. Chroma 加载参数兼容（无需修改）
        db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)
    # 使用本地api提供的大模型服务
    llm = ChatGLM(
        endpoint_url='http://127.0.0.1:8000',
        max_token=2048,
        top_p=0.9
    )
    # -------------------------- 修改：构建 QA 链（替代 RetrievalQA） --------------------------
    retriever = db.as_retriever()  # 创建检索器
    # 构建检索-生成链：检索文档 → 格式化上下文 → 拼接提示 → LLM生成 → 输出解析
    qa_chain = (
        {"context": retriever | format_docs,  # 检索并格式化文档
         "question": RunnablePassthrough()}  # 传递用户问题
        | QA_PROMPT  # 拼接提示模板
        | llm  # 调用大模型
        | StrOutputParser()  # 解析输出为字符串
    )
    # ---------------------------------------------------------------------------------------
    # 设置前端交互页面（保持不变）
    with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue=gr.themes.Color(
                    name="custom_blue",
                    c50="#e3f2fd", c100="#bbdefb", c200="#90caf9",
                    c300="#64b5f6", c400="#42a5f5", c500="#2196f3",
                    c600="#1e88e5", c700="#1976d2", c800="#1565c0", c900="#0d47a1"
                ),
                secondary_hue=gr.themes.Color(name="custom_gray", c500="#616161"),
                neutral_hue=gr.themes.Color(name="custom_light", c500="#f5f5f5")
            ),
            css="""
        /* 全局布局优化 */
        .gradio-container { max-width: 1200px !important; margin: 0 auto; padding: 2rem; }
        body { background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%); }

        /* Chatbot样式增强 */
        #AI助手 { height: 600px !important; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .gr-chatbot .user-bubble { background-color: #e3f2fd !important; color: #0d47a1 !important; border-radius: 18px !important; padding: 12px 18px !important; font-size: 16px !important; }
        .gr-chatbot .bot-bubble { background-color: #f5f5f5 !important; color: #616161 !important; border-radius: 18px !important; padding: 12px 18px !important; font-size: 16px !important; }
        .gr-chatbot .avatar { width: 40px !important; height:40px !important; border-radius:50% !important; }

        /* 输入区域样式优化 */
        .gr-textbox { border-radius: 24px !important; padding: 12px 20px !important; font-size:16px !important; border:1px solid #e0e0e0 !important; box-shadow:0 2px 6px rgba(0,0,0,0.05) !important; }
        .gr-textbox:focus { border-color: #2196f3 !important; outline:none !important; box-shadow:0 0 0 3px rgba(33,150,243,0.2) !important; }

        /* 按钮样式升级 */
        .gr-button { border-radius:24px !important; padding:8px 16px !important; font-size:16px !important; margin:0 4px !important; transition:all 0.3s ease !important; }
        .gr-button:hover { transform: translateY(-2px) !important; box-shadow:0 4px 8px rgba(0,0,0,0.1) !important; }
        .gr-upload-button { background-color:#42a5f5 !important; color:white !important; }
        .gr-upload-button:hover { background-color:#1e88e5 !important; }
        .gr-button[value*="开关知识图谱"] { background-color:#616161 !important; color:white !important; }
        .gr-button[value*="已开启"] { background-color:#2196f3 !important; }
        """
    ) as demo:
        # 状态变量：记录知识图谱是否开启
        neo4j_enabled = gr.State(False)

        # 标题与描述
        gr.Markdown("# 🤖 AI助手（知识图谱增强版）", elem_id="title")
        gr.Markdown("### 输入问题或上传文件，体验智能交互", elem_id="subtitle")

        # Chatbot组件
        chatbot = gr.Chatbot(
            [],
            elem_id="AI助手",
            bubble_full_width=False,
            avatar_images=(None, os.path.join(os.path.dirname(__file__), "bot.jpg")),
            bubbleBorderRadius=18,
            bubblePadding=12
        )

        # 输入区域布局
        with gr.Row(elem_id="input-row", variant="compact"):
            query = gr.Textbox(
                scale=5,
                show_label=False,
                placeholder="输入问题并按下回车键提交",
                container=False,
                interactive=True
            )
            btn_upload = gr.UploadButton("📁 上传外挂数据库", file_types=['txt'], elem_id="upload-btn")
            btn_neo4j = gr.Button(value="开关知识图谱（未开启）", elem_id="neo4j-btn")

        # 交互逻辑绑定
        # 1. 知识图谱开关
        btn_neo4j.click(
            fn=btn_neo4j_click,
            inputs=[chatbot, neo4j_enabled],
            outputs=[chatbot, neo4j_enabled, btn_neo4j],
            show_progress=True
        )

        # 2. 文本输入提交
        query.submit(
            fn=add_text,
            inputs=[chatbot, query],
            outputs=[chatbot, query]
        ).then(
            fn=bot,
            inputs=chatbot,
            outputs=[chatbot, query]
        )

        # 3. 文件上传处理
        btn_upload.upload(
            fn=add_file,
            inputs=[chatbot, btn_upload],
            outputs=chatbot,
            show_progress=True
        )

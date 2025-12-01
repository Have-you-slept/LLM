import time
import os
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import ChatGLM
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from neo4j import search

neo4j = False
neo4j_use = "知识图谱已启动" if neo4j else "知识图谱已关闭"


def load_documents(directory="document"):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)
    return split_docs


def load_embedding_model(local_model_path):
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cpu"}
    return HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


def add_text(history, text):
    history += [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    global qa_chain, retriever
    directory = os.path.dirname(file.name)
    documents = load_documents(directory)
    db = store_chroma(documents, embeddings)
    retriever = db.as_retriever()
    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | QA_PROMPT
            | llm
            | StrOutputParser()
    )
    history = history + [((file.name,), None)]
    return history


### 修复：bot函数返回两个值
def bot(history):
    global neo4j, neo4j_use, qa_chain
    if not history:
        yield history, gr.update(interactive=True)  # 初始返回两个值
        return
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
        response = qa_chain.invoke(message)
        response += extra
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        ### 每次yield返回两个值：chatbot + 恢复输入框交互
        yield history, gr.update(interactive=True)


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


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


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

if __name__ == "__main__":
    embeddings = load_embedding_model(local_model_path=r'D:\simple_RAG_with_LLMs_API\text2vec-base-chinese')
    if not os.path.exists('VectorStore'):
        documents = load_documents()
        db = store_chroma(documents, embeddings)
    else:
        db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)
    llm = ChatGLM(
        endpoint_url='http://127.0.0.1:8000',
        max_token=2048,
        top_p=0.9
    )
    retriever = db.as_retriever()
    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | QA_PROMPT
            | llm
            | StrOutputParser()
    )
    with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue=gr.themes.Color(
                    name="custom_blue",
                    c50="#e3f2fd", c100="#bbdefb", c200="#90caf9",
                    c300="#64b5f6", c400="#42a5f5", c500="#2196f3",
                    c600="#1e88e5", c700="#1976d2", c800="#1565c0",
                    c900="#0d47a1", c950="#0a3d62"
                ),
                secondary_hue=gr.themes.Color(
                    name="custom_gray",
                    c50="#fafafa", c100="#f5f5f5", c200="#eeeeee",
                    c300="#e0e0e0", c400="#bdbdbd", c500="#616161",
                    c600="#424242", c700="#333333", c800="#212121",
                    c900="#000000", c950="#0a0a0a"
                ),
                neutral_hue=gr.themes.Color(
                    name="custom_light",
                    c50="#ffffff", c100="#f9f9f9", c200="#f0f0f0",
                    c300="#e6e6e6", c400="#dcdcdc", c500="#f5f5f5",
                    c600="#cccccc", c700="#b3b3b3", c800="#999999",
                    c900="#7f7f7f", c950="#666666"
                )
            ),
            css="""
        .gradio-container { max-width:1200px !important; margin:0 auto; padding:2rem; }
        body { background:linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%); }
        #AI助手 { height:600px !important; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1); }
        .gr-chatbot .user-bubble { background:#e3f2fd !important; color:#0d47a1 !important; border-radius:18px !important; padding:12px 18px !important; font-size:16px !important; }
        .gr-chatbot .bot-bubble { background:#f5f5f5 !important; color:#616161 !important; border-radius:18px !important; padding:12px 18px !important; font-size:16px !important; }
        .gr-chatbot .avatar { width:40px !important; height:40px !important; border-radius:50% !important; }
        .gr-textbox { border-radius:24px !important; padding:12px 20px !important; font-size:16px !important; border:1px solid #e0e0e0 !important; box-shadow:0 2px 6px rgba(0,0,0,0.05) !important; }
        .gr-textbox:focus { border-color:#2196f3 !important; outline:none !important; box-shadow:0 0 0 3px rgba(33,150,243,0.2) !important; }
        .gr-button { border-radius:24px !important; padding:8px 16px !important; font-size:16px !important; margin:0 4px !important; transition:all 0.3s ease !important; }
        .gr-button:hover { transform:translateY(-2px) !important; box-shadow:0 4px 8px rgba(0,0,0,0.1) !important; }
        .gr-upload-button { background:#42a5f5 !important; color:white !important; }
        .gr-upload-button:hover { background:#1e88e5 !important; }
        .gr-button[value*="开关知识图谱"] { background:#616161 !important; color:white !important; }
        .gr-button[value*="已开启"] { background:#2196f3 !important; }
        """
    ) as demo:
        neo4j_enabled = gr.State(False)
        gr.Markdown("# 🤖 AI助手（知识图谱增强版）", elem_id="title")
        gr.Markdown("### 输入问题或上传文件，体验智能交互", elem_id="subtitle")

        chatbot = gr.Chatbot(
            [], elem_id="AI助手", bubble_full_width=False,
            avatar_images=(None, os.path.join(os.path.dirname(__file__), "bot.jpg")),
        )

        with gr.Row(elem_id="input-row", variant="compact"):
            query = gr.Textbox(scale=5, show_label=False, placeholder="输入问题并按下回车键提交", container=False,
                               interactive=True)
            btn_upload = gr.UploadButton("📁 上传外挂数据库", file_types=['txt'], elem_id="upload-btn")
            btn_neo4j = gr.Button(value="开关知识图谱（未开启）", elem_id="neo4j-btn")

        btn_neo4j.click(fn=btn_neo4j_click, inputs=[chatbot, neo4j_enabled],
                        outputs=[chatbot, neo4j_enabled, btn_neo4j], show_progress=True)
        query.submit(fn=add_text, inputs=[chatbot, query], outputs=[chatbot, query]).then(fn=bot, inputs=chatbot,
                                                                                          outputs=[chatbot, query])
        btn_upload.upload(fn=add_file, inputs=[chatbot, btn_upload], outputs=chatbot, show_progress=True)

    demo.launch(debug=True, server_name="0.0.0.0", server_port=7860)
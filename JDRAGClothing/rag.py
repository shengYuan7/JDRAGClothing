import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_core.documents import Document  # 正确的 Document 导入
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

# 加载环境变量
load_dotenv()

def print_prompt(prompt):
    """打印提示模板内容，用于调试"""
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(
                model=config.embedding_model_name,
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
            ),
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                           "简洁和专业的回答用户问题。参考资料：{context}。"),
                ("system", "并且我提供用户的对话历史记录,如下: "),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(
            model=config.chat_model_name,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        """构建 LCEL 链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            """将检索到的文档列表格式化为字符串"""
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
                {
                    "input": RunnablePassthrough(),  # 实例化以接收用户输入
                    "context": RunnableLambda(format_for_retriever) | retriever | format_document  # 检索并格式化文档
                }
                | RunnableLambda(format_for_prompt_template)
                | self.prompt_template
                | print_prompt  # 可选调试步骤
                | self.chat_model
                | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == "__main__":
    # session id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = RagService().chain.invoke({"input": "冬天穿什么颜色?"}, session_config)
    print(res)

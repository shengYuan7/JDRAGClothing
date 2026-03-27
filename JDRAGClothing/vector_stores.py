import os  # 添加 os 模块用于环境变量
from dotenv import load_dotenv
from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器,方便加入chain"""
        # 修正：将 search_kwargs 写成正确的字典格式（原代码误用集合）
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})


if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings

    # 加载环境变量
    load_dotenv()

    retriever = VectorStoreService(
        DashScopeEmbeddings(
            model=config.embedding_model_name,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")  # 请确保该参数名与库兼容
        )
    ).get_retriever()

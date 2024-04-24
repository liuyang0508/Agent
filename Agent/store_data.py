import os
import sys

current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
os.chdir(root_dir)

import asyncio
import aiofiles

import numpy as np
import pickle

from Agent.vector_process_by_faiss import VectorIndexManager

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


class StoreData:

    @staticmethod
    def data_loader_web(url):
        """
        数据加载

        Parameters:
        url (str): 数据的来源 URL

        Returns:
        list: 加载的数据列表
        """
        return WebBaseLoader(url).load()

    @staticmethod
    async def store_data(data: list, model_name: str = "BAAI/bge-reranker-large", metric: str = 'L2',
                   vectorized_texts_file: str = "vectorized_texts.pkl", text_index_map_file: str = "text_index_map.pkl"):
        """
        数据存储

        Parameters:
        data (list): 需要存储的数据列表
        model_name (str): 模型名称，默认为 "BAAI/bge-reranker-large"
        metric (str): 索引类型，默认为 'L2'
        vectorized_texts_file (str): 存储向量化文本的文件名，默认为 "vectorized_texts.pkl"
        text_index_map_file (str): 存储索引映射关系的文件名，默认为 "text_index_map.pkl"
        """

        # 分块
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=5)
        texts = text_splitter.split_documents(data)

        # 向量化
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorized_texts = []
        texts_with_index = []

        for doc in texts:
            text_content = doc.page_content
            vectorized_text = embeddings.embed_query(text_content)
            vectorized_texts.append(vectorized_text)
            texts_with_index.append((doc, vectorized_text))

        # 创建 VectorIndexManager 实例
        index_manager = VectorIndexManager(dimension=len(vectorized_texts[0]), metric=metric)
        index_manager.initialize_index(np.stack(vectorized_texts))

        # 添加向量到索引
        index_manager.add_vectors(np.stack(vectorized_texts))

        # 构建索引与原始文本的映射关系
        text_index_map = {tuple(vectorized_text): text for text, vectorized_text in texts_with_index}

        # 将 vectorized_texts 和 text_index_map 存储到文件
        async with aiofiles.open(vectorized_texts_file, mode="wb") as f:
            await f.write(pickle.dumps(vectorized_texts))

        async with aiofiles.open(text_index_map_file, mode="wb") as f:
            await f.write(pickle.dumps(text_index_map))


if __name__ == "__main__":
    url = "https://zhuanlan.zhihu.com/p/40487710"
    
    async def main():
        await StoreData.store_data(StoreData.data_loader_web(url))

    asyncio.run(main())
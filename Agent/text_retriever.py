import os
import sys

current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
os.chdir(root_dir)

import asyncio

import numpy as np
import pickle

from Agent.query_order_dispatch_strategy import fetch_nacos_config

from Agent.vector_process_by_faiss import VectorIndexManager

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import KNNRetriever
from langchain.memory import ConversationBufferWindowMemory


class TextRetriever:

    @staticmethod
    def load_index_and_map(vectorized_texts_file: str = "vectorized_texts.pkl", text_index_map_file: str = "text_index_map.pkl"):
        # 从文件加载 vectorized_texts 和 text_index_map 数据
        with open(vectorized_texts_file, "rb") as f:
            vectorized_texts = pickle.load(f)

        with open(text_index_map_file, "rb") as f:
            text_index_map = pickle.load(f)
        return vectorized_texts, text_index_map
    
    @staticmethod
    async def text_retriever(queries: str, model_name: str = "BAAI/bge-reranker-large", metric: str = 'L2', k: int = 1):
        # RAG
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        query_vectors = [vec for vec in embeddings.embed_query(queries)]
        vectorized_queries = np.stack(query_vectors)

        index_managers = VectorIndexManager(dimension=len(embeddings.embed_query(queries)), metric=metric)
        indices, _ = index_managers.search_vectors(vectorized_queries, k=k)

        # 从文件加载 vectorized_texts 和 text_index_map 数据
        vectorized_texts, text_index_map = TextRetriever.load_index_and_map()

        # 获取查询结果对应的原始文本
        all_results = []
        for query_indices in indices:
            query_results = [text_index_map[tuple(vectorized_texts[i])] for i in query_indices]
            all_results.append(query_results)

        return all_results
            
    @staticmethod
    async def fetch_nacos_config():
        return fetch_nacos_config('http://nacos-dev.hzcxfw.com:8848', 'itbox-nacos-dev', 'DIQ-matching-vehicle-service.yaml', 'DIQ')


if __name__ == "__main__":
    queries = "'向量数据库解决的问题从技术角度来讲，向量数据库主要解决2个问题，一个是高效的检索，另一个是高效的分析。1）检索通常就是图片检索图片，例如人脸检索，人体检索，和车辆检索，以及猫厂的商品图片检索，人脸支付。2）分析在平安城市应用的比较多，例如人脸撞库，公安会把2个类似作案手法的案发现场周边的人像做对比，看哪些人同时在2个案发现场出现。随着国家安全和反恐的需求增长，根据业务规划，深圳平安城市项目到2018年底，会部署20w摄像头，预计保留一年的人脸特征在千亿级别；以及人们对购物体验的提升，商品种类以亿计，后续还可以支持音频和非结构化的文本检索，向量数据库大有可为。3.'"
    
    async def main():
        results = await TextRetriever.text_retriever(queries)
        print("answer: {}", results)

    asyncio.run(main())
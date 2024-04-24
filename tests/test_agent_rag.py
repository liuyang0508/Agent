import os
import sys

current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
os.chdir(root_dir)

from Agent.store_data import StoreData
from Agent.text_retriever import TextRetriever

import asyncio


async def test_agent_rag():
    # data process
    url = "https://zhuanlan.zhihu.com/p/40487710"
    await StoreData.store_data(StoreData.data_loader_web(url))

    # RAG
    queries = "'向量数据库解决的问题从技术角度来讲，向量数据库主要解决2个问题，一个是高效的检索，另一个是高效的分析。1）检索通常就是图片检索图片，例如人脸检索，人体检索，和车辆检索，以及猫厂的商品图片检索，人脸支付。2）分析在平安城市应用的比较多，例如人脸撞库，公安会把2个类似作案手法的案发现场周边的人像做对比，看哪些人同时在2个案发现场出现。随着国家安全和反恐的需求增长，根据业务规划，深圳平安城市项目到2018年底，会部署20w摄像头，预计保留一年的人脸特征在千亿级别；以及人们对购物体验的提升，商品种类以亿计，后续还可以支持音频和非结构化的文本检索，向量数据库大有可为。3.'"
    results =  await TextRetriever.text_retriever(queries)
    print("answer: {}", results)
    

# Run the test
if __name__ == "__main__":
    async def main():
        await test_agent_rag()

    asyncio.run(main())

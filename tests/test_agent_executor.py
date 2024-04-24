import os
import sys
import asyncio
import yaml

current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
os.chdir(root_dir)

from Agent.store_data import StoreData
from Agent.text_retriever import TextRetriever

# ANSI颜色代码
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

async def test_agent_rag():
    # data process
    url = "https://zhuanlan.zhihu.com/p/40487710"
    await StoreData.store_data(StoreData.data_loader_web(url))

    while True:
        # RAG
        queries = input(f"{YELLOW}请输入查询内容：{RESET}")
        if "查看策略配置" in queries:
            config = await TextRetriever.fetch_nacos_config()
            if config is not None:
                print(f"{GREEN}answer: {yaml.dump(config, default_flow_style=False)}{RESET}")
            else:
                print(f"{RED}answer: 获取或解析策略配置失败{RESET}")
        else:
            results = await TextRetriever.text_retriever(queries)
            print(f"{GREEN}answer: {results}{RESET}")

        another_question = input(f"{BLUE}是否还有其他问题？(yes/no): {RESET}")
        if another_question.lower() != 'yes':
            break

# Run the test
if __name__ == "__main__":
    async def main():
        await test_agent_rag()

    asyncio.run(main())

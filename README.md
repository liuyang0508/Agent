## 构建 Docker 镜像
> docker build -t agent-executor .

![alt text](datas/images/构建Docker镜像.png)

## 运行 Docker 容器
> docker run -it --network=host agent-executor bash

## 运行 RAG 脚本
> python tests/test_agent_executor.py 

![alt text](datas/images/运行Docker容器.png)

![alt text](datas/images/运行RAG脚本.png)
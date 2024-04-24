## 构建 Docker 镜像
> docker build -t agent-executor .

![alt text](image.png)

## 运行 Docker 容器
> docker run -it --network=host agent-executor bash

## 运行 RAG 脚本
> python tests/test_agent_executor.py 

![alt text](image-1.png)

![alt text](image-2.png)
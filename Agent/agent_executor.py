import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import KNNRetriever
from langchain.memory import ConversationBufferWindowMemory

def initialize_vector_index(urls, chunk_size=500, chunk_overlap=10, index_file='vector_index.bin'):
    """
    初始化向量索引。仅在首次运行或需要更新索引时调用。

    Args:
        urls (List[str]): 待加载的文档URL列表。
        chunk_size (int): 文本分块大小。
        chunk_overlap (int): 文本块之间的重叠字符数。
        index_file (str): 索引文件路径。

    Returns:
        None
    """
    # 加载数据
    all_data = []
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()
        all_data.extend(data)

    # 分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(all_data)

    # 向量化
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    db.save(index_file)  # 保存索引到文件


def search_vector_index(query, index_file='vector_index.bin', top_k=10):
    """
    使用已保存的向量索引来检索与查询相关的内容。

    Args:
        query (str): 待查询的问题或关键词。
        index_file (str): 索引文件路径。
        top_k (int): 返回结果的前k个最相关文档。

    Returns:
        List[Tuple[float, str]]: 包含相似度分数和对应文本块的元组列表，按相似度降序排列。
    """
    db = FAISS.load(index_file)  # 从文件加载已有索引
    retriever = db.as_retriever()
    result = retriever.get_relevant_documents(query, k=top_k)
    return result


# 首次运行或需要更新索引时，调用此函数
if not os.path.exists('vector_index.bin') or os.getenv('REBUILD_INDEX', 'false').lower() == 'true':
    urls = [
        "https://zhuanlan.zhihu.com/p/659386520",
        "https://zhuanlan.zhihu.com/p/692795508",
    ]
    chunk_size = 500
    chunk_overlap = 10
    initialize_vector_index(urls, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# 在智能问答场景中，只需调用search_vector_index函数进行检索
query = "包含"
results = search_vector_index(query)
print(results)
# 1 chroma
# 安装 pip install chromadb -i https://pypi.tuna.tsinghua.edu.cn/simple
# 运行 chroma run --path ./data
# 相关命令
#   --path            TEXT     The path to the file or directory. [default: ./chroma_data]                                                                                                                                                  │
#   --host            TEXT     The host to listen to. Default: localhost [default: localhost]                                                                                                                                               │
#   --log-path        TEXT     The path to the log file. [default: chroma.log]                                                                                                                                                              │
#   --port            INTEGER  The port to run the server on. [default: 8000]                                                                                                                                                               │
#  --help                     Show this message and exit.

# import chromadb

# chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# 使用chroma中的模型函数
# from chromadb.utils import embedding_functions

# model_path = "./models/gte-large-zh/"

# em_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_path)

# collection = chroma_client.create_collection(
#     name="rag_db",
#     embedding_function=em_fn,
#     metadata={"hnsw:space": "cosine"}  # hnsw是chroma的默认索引,也只支持这一种索引. 设置检索模式为余弦相似度
# )

# documents = ["在向量搜索领域，我们拥有多种索引方法和向量处理技术，\
#     它们使我们能够在召回率、响应时间和内存使用之间做出权衡。",
#              "虽然单独使用特定技术如倒排文件（IVF）、乘积量化（PQ）\
#              或分层导航小世界（HNSW）通常能够带来满意的结果",
#              "GraphRAG 本质上就是 RAG，只不过与一般 RAG 相比，其检索路径上多了一个知识图谱"]

# collection.add(
#     documents=documents,
#     ids=["id1", "id2", "id3"],
#     metadatas=[{"chapter": 3, "verse": 16},
#                {"chapter": 4, "verse": 5},
#                {"chapter": 12, "verse": 5}]
# )

# ret = collection.count()
# print(ret)
# ret = collection.peek(limit=1)
# print(ret)

# 进行检索
# get_collection = chroma_client.get_collection(
#     name="rag_db",
#     embedding_function=em_fn,
# )

# id_ret = get_collection.get(
#     ids=["id2"],
#     include=["documents", "embeddings", "metadatas"]
# )

# print(id_ret)
# print(id_ret['documents'])
# print(id_ret['metadatas'])

# import numpy as np

# ret = np.array(id_ret['embeddings']).shape
# print(ret)

# query = "索引技术有哪些?"
# ret = get_collection.query(
#     query_texts=query,
#     n_results=2,
#     include=["documents", "metadatas"],
#     where={'verse': 5}  # 进行元数据过滤,首先
# )
# print(ret)

# 混合检索支持的操作
# - $eq - equal to (string, int, float)
# - $ne - not equal to (string, int, float)
# - $gt - greater than (int, float)
# - $gte - greater than or equal to (int, float)
# - $lt - less than (int, float)
# - $lte - less than or equal to (int, float)

# ret = get_collection.query(
#     query_texts=["索引技术有哪些？"],
#     n_results=2,
#     where={"$and": [{"chapter": {"$lt": 10}},
#                     {"verse": {"$eq": 5}}
#                     ]}
# )
# print(ret)

# ret = get_collection.query(
#     query_texts=["索引技术有哪些？"],
#     n_results=2,
#     where_document={"$contains": "索引"}  # 文档过滤,包含了"索引"这两个字的文档
# )
# print(ret)

# 2 milvus
# 通过docker-compose: wget https://github.com/milvus-io/milvus/releases/download/v2.2.8/milvus-standalone-docker-compose.yml -O docker-compose.yml
# 注意: 版本要匹配 milvus和pymilvus

# import numpy as np
# from huggingface_hub import get_collection
# from pymilvus import (
#     connections,
#     utility,
#     FieldSchema,
#     CollectionSchema,
#     DataType,
#     Collection
# )

# connections.connect(host="localhost", port="19530")

# fileds = [
#     FieldSchema(name="pk", dtype=DataType.VARCHAR,
#                 is_primary=True, auto_id=False, max_length=100),
#     FieldSchema(name="documents", dtype=DataType.VARCHAR, max_length=512),
#     FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024),
#     FieldSchema(name="verse", dtype=DataType.INT64),
# ]

# rag_db = Collection(
#     "rag_db",
#     CollectionSchema(fileds),
#     consistency_level="Strong"  # 数据强一致性标识
# )

# documents = ["在向量搜索领域，我们拥有多种索引方法和向量处理技术，\
#     它们使我们能够在召回率、响应时间和内存使用之间做出权衡。",
#              "虽然单独使用特定技术如倒排文件（IVF）、乘积量化（PQ）\
#              或分层导航小世界（HNSW）通常能够带来满意的结果",
#              "GraphRAG 本质上就是 RAG，只不过与一般 RAG 相比，其检索路径上多了一个知识图谱"]

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# model_path = "./models/gte-large-zh/"
# model = HuggingFaceEmbeddings(model_name=model_path,
#                               model_kwargs={'device': "cpu"})
# embeddings = model.embed_documents(documents)

# 插入数据
# entities = [
#     [str(i) for i in range(len(documents))],
#     documents,
#     np.array(embeddings),
#     [16, 5, 5],
# ]

# insert_ret = rag_db.insert(entities)
# rag_db.flush()
# ret = rag_db.num_entities
# print(ret)

# 创建索引
# index = {
#     "index_type": "IVF_FLAT",  # 使用倒排索引
#     "metric_type": "L2",
#     "params": {"nlist": 128},
# }
# rag_db.create_index("embeddings", index)

# 检索
# find_collection = Collection("rag_db")
# find_collection.load()

# query = "索引技术有哪些?"
# query_emb = model.embed_documents([query])

# ret = find_collection.search(
#     query_emb,
#     "embeddings",
#     param={"metric_type": "L2"},  # 度量方式, L2的欧式距离度量
#     limit=2,
#     output_fields=["documents", "verse"],
# )
# print(ret)

# for hits in ret:
#     for hit in hits:
#         print(f"hit: {hit}, documents field: {hit.entity.get('documents')}")

# result2 = get_collection.search(query_emb,
#                                 "embeddings",
#                                 param={"metric_type": "L2"},
#                                 expr="verse < 10",
#                                 limit=1,
#                                 output_fields=["documents", "verse"])

# for hits in result2:
#     for hit in hits:
#         print(f"hit: {hit}, documents field: {hit.entity.get('documents')}")

# 3 index索引优化
import numpy as np
from scipy.cluster.vq import kmeans2

# 随机生成一个查询向量（128维）
query = np.random.normal(size=(128,))
# 随机生成一个包含1000个向量的数据集（每个向量128维）

# 暴力搜索：计算查询向量与数据集中所有向量的欧氏距离
# np.linalg.norm(query - dataset, axis=1) 计算了query与dataset中每个向量的距离
# np.argmin() 找到距离最小的那个向量的索引
dataset = np.random.normal(size=(1000, 128))
ret = np.argmin(np.linalg.norm(query - dataset, axis=1))
print(ret)

# IVF 倒排索引
num_part = 100
# 使用 k-means 算法将数据集分成 100 个簇
# centroids: 100个簇的中心点
# assignments: 数据集中每个向量所属的簇ID
(centroids, assignments) = kmeans2(dataset, num_part, iter=1000)
print(centroids.shape)
print(assignments[:10])

# 创建倒排索引：一个列表，每个元素是一个簇，里面存储属于该簇的所有向量的索引
index = [[] for _ in range(num_part)]
for n, k in enumerate(assignments):
    index[k].append(n)
print(index[1])

# 第一步：粗略查找 - 找到查询向量最可能属于的簇
cluster_id = np.argmin(np.linalg.norm(query - centroids, axis=1))
print(cluster_id)

# 第二步：精细查找 - 只在找到的那个簇中搜索
ret = np.argmin(np.linalg.norm(query - dataset[index[cluster_id]], axis=1))
print(ret)

# 为了提高准确性（召回率），搜索多个最近的簇
# 找到离查询向量最近的3个簇
cluster_ids = np.argsort(np.linalg.norm(query - centroids, axis=1))[:3]
print(cluster_ids)

# 将这3个簇中的所有向量索引合并成一个列表
top3_index = []
for c in cluster_ids:
    top3_index += index[c]

# 在合并后的所有向量中进行搜索
ret = np.argmin(np.linalg.norm(query - dataset[top3_index], axis=1))
print(ret)

# 由于之前 ret 是在 top3_index 这个局部列表中的索引，我们还需要找到它在原始 dataset 中的索引
# 比如 ret = 43，那么 top3_index[43] 就是其在原始数据集中的索引
ret = top3_index[43]
print(ret)

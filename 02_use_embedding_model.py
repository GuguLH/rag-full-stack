# 3种embedding调用方式:transformers和sentence_transformers、langchain

# 1 transformers 方式
# from transformers import AutoTokenizer, AutoModel
# from sentence_transformers.util import cos_sim
# import torch.nn.functional as f
#
# input_texts = [
#     "中国的首都是哪里",
#     "你喜欢去哪里旅游",
#     "北京",
#     "今天中午吃什么"
# ]
#
# model_path = "./models/gte-large-zh/"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path, device_map="cpu")
# batch_tokens = tokenizer(
#     input_texts,
#     max_length=30,
#     padding=True,  # 按照最长的句子补齐
#     truncation=True,
#     return_tensors="pt"
# )
# ret = batch_tokens[0].tokens
# print(ret)
# ret = batch_tokens[2].tokens
# print(ret)
# ret = batch_tokens.input_ids[0]
# print(ret)
# ret = batch_tokens.input_ids[2]
# print(ret)
#
# outputs = model(**batch_tokens)
# print(outputs)
# ret = outputs.last_hidden_state.shape
# print(ret)
# ret = outputs.pooler_output.shape
# print(ret)
#
# embeddings = outputs.last_hidden_state[:, 0]
# print(embeddings.shape)
#
# embeddings = f.normalize(embeddings, p=2, dim=-1)
# for i in range(1, 4):
#     print(input_texts[0], input_texts[i], cos_sim(embeddings[0], embeddings[i]))

# 2 sentence_transformers 方式
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim
#
# input_texts = [
#     "中国的首都是哪里",
#     "你喜欢去哪里旅游",
#     "北京",
#     "今天中午吃什么"
# ]
#
# model_path = "./models/gte-large-zh/"
#
# model = SentenceTransformer(model_path)
# embeddings = model.encode(input_texts)
# ret = embeddings.shape
# print(ret)
#
# for i in range(1, 4):
#     print(input_texts[0], input_texts[i], cos_sim(embeddings[0], embeddings[i]))

# 3 langchain 方式
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sentence_transformers.util import cos_sim

model_path = "./models/gte-large-zh/"

model = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={"device": "cpu"}
)

input_texts = [
    "中国的首都是哪里",
    "你喜欢去哪里旅游",
    "北京",
    "今天中午吃什么"
]

embeddings = model.embed_documents(input_texts)

import numpy as np

embeddings = np.array(embeddings)
print(embeddings.shape)

for i in range(1, 4):
    print(input_texts[0], input_texts[i], cos_sim(embeddings[0], embeddings[i]))

# 4 embedding操作
# 余弦相似度
a = embeddings[0]
b = embeddings[2]
from numpy import dot
from numpy.linalg import norm

cos_a_b = dot(a, b) / (norm(a) * norm(b))
print(cos_a_b, cos_sim(a, b))

# 欧几里得距离: 如果两个向量越相似,欧式距离越小;完全相似为0
ret = norm(a - b)
print(ret)

# 聚类
texts = ['苹果', '菠萝', '西瓜', '斑马', '大象', '老鼠']
output_embeddings = model.embed_documents(texts)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(output_embeddings)
label = kmeans.labels_
for i in range(len(texts)):
    print(f"cls({texts[i]}) = {label[i]}")

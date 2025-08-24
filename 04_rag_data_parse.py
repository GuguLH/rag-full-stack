# 1 langchain document格式
# from langchain_core.documents import Document

# document = Document(
#     page_content="hello, world",
#     metadata={"source": "test meta data"}
# )
# ret = document.page_content
# print(ret)
# ret = document.metadata
# print(ret)

# 2 html
# from langchain_community.document_loaders import WebBaseLoader

# loader = WebBaseLoader("https://flask.palletsprojects.com/en/3.0.x/tutorial/layout/")
# docs = loader.load()
# for doc in docs:
#     print(doc.page_content, doc.metadata)

# from langchain_community.document_loaders import BSHTMLLoader

# loader = BSHTMLLoader("./files/test.html", open_encoding="utf-8")
# data = loader.load()
# for doc in data:
#     print(doc.page_content, doc.metadata)

# from bs4 import BeautifulSoup

# html_txt = ""
# with open("./files/test.html", "r", encoding="utf-8") as f:
#     for line in f.readlines():
#         html_txt += line

# soup = BeautifulSoup(html_txt, "lxml")
# code_content = soup.find_all("div", class_="highlight")
# for ele in code_content:
#     print(ele.text)
#     print("+" * 100)

# 3 pdf
# from langchain_community.document_loaders import PyMuPDFLoader

# loader = PyMuPDFLoader("./files/zhidu_travel.pdf")
# docs = loader.load()
# for doc in docs:
#     print(doc)
#     print("=" * 100)

# 使用fitz加载pdf表格
# import fitz

# doc = fitz.open("./files/zhidu_travel.pdf")
# table_data = []
# text_data = []

# doc_tables = {}
# for idx, page in enumerate(doc):
#     text = page.get_text()
#     text_data.append(text)
#     tabs = page.find_tables()
#     for i, tab in enumerate(tabs):
#         ds = tab.to_pandas()
#         table_data.append(ds.to_markdown())

# for tab in table_data:
#     print(tab)
#     print("=" * 100)

# 4 Unstructured
# 导入所需的库
# from langchain_unstructured import UnstructuredLoader

# print("开始加载PDF文件...")
# print("=" * 100)

# 使用 Unstructured Loader 加载 PDF 文件
# loader = UnstructuredLoader("./files/zhidu_travel.pdf")
# docs = loader.load()

# 打印处理后的文档内容
# for doc in docs:
#     print(doc)
#     print("=" * 100)

# 5 ppt word excel
# from langchain_community.document_loaders import UnstructuredPowerPointLoader

# loader = UnstructuredPowerPointLoader("./files/test_ppt.pptx")
# docs = loader.load()
# for doc in docs:
#     print(doc)
#     print("=" * 100)

# from pptx import Presentation
# from pptx.enum.shapes import MSO_SHAPE_TYPE

# ppt = Presentation("./files/test_ppt.pptx")
# for slide_number, slide in enumerate(ppt.slides, start=1):
#     print(f"Slide {slide_number}:")
#     for shape in slide.shapes:
#         if shape.has_text_frame:
#             print(shape.text)
#
#         if shape.has_table:
#             table = shape.table
#             for row_idx, row in enumerate(table.rows):
#                 for col_idx, cell in enumerate(row.cells):
#                     cell_text = cell.text
#                     print(f"Row {row_idx + 1}, Column {col_idx + 1}: {cell_text}")
#
#         if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
#             image = shape.image
#             image_filename = './files/test.jpg'
#             with open(image_filename, 'wb') as f:
#                 f.write(image.blob)

# word
# from langchain_community.document_loaders import Docx2txtLoader

# loader = Docx2txtLoader("./files/test_word.docx")
# data = loader.load()

# for doc in data:
#     print(doc)
#     print("=" * 100)

# from docx import Document


# def read_docx(file_path):
#     doc = Document(file_path)
#     for para in doc.paragraphs:
#         print(para.text)
#
#     for table in doc.tables:
#         for row in table.rows:
#             for cell in row.cells:
#                 print(cell.text, end=' | ')
#             print()


# 使用示例
# file_path = './files/test_word.docx'
# read_docx(file_path)

# excel
# from openpyxl import load_workbook
#
# wb = load_workbook("./files/detail.xlsx")
# ws = wb.active
#
# for row in ws.iter_rows():
#     for cell in row:
#         print(cell.value, cell.coordinate)
#     break
#
# for merged_range in ws.merged_cells.ranges:
#     value = ws.cell(row=merged_range.min_row, column=merged_range.min_col).value
#     print(merged_range, value)
#     break

# import openpyxl
#
# workbook = openpyxl.load_workbook('./files/detail.xlsx')
# sheet = workbook.active
#
# for row in sheet.iter_rows():
#     info = []
#     for cell in row:
#         cell_value = cell.value
#         info.append(cell_value)
#     print(len(info), info)

# 6 ragflow:deepdoc
# from deepdoc_parse import Pdf, chunk


# def dummy(prog=None, msg=""):
#     print(prog, msg)


# res = chunk('./files/zhidu_travel.pdf', callback=dummy)
# for data in res:
#     print("=" * 10)
#     print(data['content_with_weight'])

# 7 文档分块
# import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker

doc_path = './files/zhidu_travel.pdf'
loader = PyMuPDFLoader(doc_path)
docs = loader.load()

# data = []
# for doc in docs:
#     data += doc.page_content.split('\u3002')

# word_nums = [len(txt) for txt in data if len(txt) != 0]
# ret = np.mean(word_nums)
# print(ret)
# ret = np.min(word_nums)
# print(ret)
# ret = np.max(word_nums)
# print(ret)
# ret = np.percentile(word_nums, q=95)
# print(ret)

# 递归文本分块：RecursiveCharacterTextSplitter
# r_spliter = RecursiveCharacterTextSplitter(
#     chunk_size=128,  # 可以比最大的长度大一点
#     chunk_overlap=30,  # 可以比平均的55小一点
#     separators=["\n\n",
#                 "\n",
#                 ".",
#                 "\uff0e",  # Fullwidth full stop
#                 "\u3002",  # Ideographic full stop
#                 ",",
#                 "\uff0c",  # Fullwidth comma
#                 "\u3001",  # Ideographic comma
#                 ]
# )

# split_docs = loader.load_and_split(r_spliter)
# print(split_docs[0].page_content)
# print(split_docs[1].page_content)

# 先进行加载,再进行分块
# docs = loader.load()
# split_docs = r_spliter.split_documents(docs)
# print(split_docs[0].page_content)

# 基于向量的语义分块：SemanticChunker
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_experimental.text_splitter import SemanticChunker

# model_path = "./models/gte-large-zh/"
# embeddings = HuggingFaceEmbeddings(
#     model_name=model_path,
#     model_kwargs={'device': "cpu"}
# )

# semantic_chunker = SemanticChunker(
#     embeddings,
#     breakpoint_threshold_type="percentile",  # 分块阈值策略,使用百分位
#     sentence_split_regex='(?<=[。；：])',  # 句子分割符
#     buffer_size=2  # 滑动窗口大小
# )

# semantic_docs = semantic_chunker.create_documents([docs[0].page_content])
# for semantic_chunk in semantic_docs:
#     print(semantic_chunk.page_content)
#     print(len(semantic_chunk.page_content))
#     print("=" * 100)

# 基于模型的语义分块：nlp_bert_document-segmentation
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    task=Tasks.document_segmentation,
    model='damo/nlp_bert_document-segmentation_chinese-base'
)

result = p(documents=docs[1].page_content.replace('\n', ''))
print(result['text'])

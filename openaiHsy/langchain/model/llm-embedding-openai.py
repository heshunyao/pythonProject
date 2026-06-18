# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
print(embeddings.embed_query("Hello world"))


files = ["data/a.txt", "data/b.txt"]
for file in files:
    with open(file, "r",  encoding="utf-8") as f:
        doc = f.read()
        doc_result = embeddings.embed_documents(doc)
        print(doc_result)

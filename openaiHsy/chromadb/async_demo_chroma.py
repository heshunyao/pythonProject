# import asyncio
# import chromadb
#
#
# async def query_collection(collection, query_str, n):
#     result = await collection.query(
#         query_texts=[query_str],
#         n_results=n
#     )
#     return result
#
#
# async def main():
#     client = await chromadb.AsyncHttpClient(host="localhost", port=8000)
#
#     collection = await client.get_or_create_collection(name="db_path")
#     await collection.add(
#         documents=["这是关于一篇java开发的文档", "这是关于一篇游戏用法的文档"],
#         ids=["id1", "id2"]
#     )
#     print("---------")
#     query = "java"
#
#     result = await query_collection(collection, query, 2)
#
#     print(result)
#
# if __name__ == "__main__":
#     asyncio.run(main())



import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection(name="db_path")
collection.add(
    documents=["这是关于一篇java开发的文档", "这是关于一篇游戏用法的文档"],
    ids=["id1", "id2"]
)
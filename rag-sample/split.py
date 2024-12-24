import os
import time
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


def read_document_content(filename):
    current_path = os.getcwd()
    print(f"Current path: {current_path}")
    with open(filename, "r") as f:
        return f.read()


# Chunk the document based on h2 headers.
markdown_document = read_document_content("./rag-sample/sample-product.md")

headers_to_split_on = [
    ("##", "Header 2")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)

print(md_header_splits)
print("\n")

model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY')
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "rag-getting-started"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec=spec
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# See that it is empty
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")

namespace = "wondervector5000"

docsearch = PineconeVectorStore.from_documents(
    documents=md_header_splits,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

time.sleep(5)

# See how many vectors have been upserted
print("Index after upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")
time.sleep(2)

index = pc.Index(index_name)
namespace = "wondervector5000"

for ids in index.list(namespace=namespace):
    query = index.query(
        id=ids[0],
        namespace=namespace,
        top_k=1,
        include_values=True,
        include_metadata=True
    )
    print(query)
    print("\n")

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = docsearch.as_retriever()

llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

query1 = "What are the first 3 steps for getting started with the WonderVector5000?"
query2 = "The Neural Fandango Synchronizer is giving me a headache. What do I do?"
answer1_without_knowledge = llm.invoke(query1)

print("Query 1:", query1)
print("\nAnswer without knowledge:\n\n", answer1_without_knowledge.content)
print("\n")
time.sleep(2)

answer1_with_knowledge = retrieval_chain.invoke({"input": query1})

print("Answer with knowledge:\n\n", answer1_with_knowledge['answer'])
print("\nContext used:\n\n", answer1_with_knowledge['context'])
print("\n")
time.sleep(2)

answer2_without_knowledge = llm.invoke(query2)

print("Query 2:", query2)
print("\nAnswer without knowledge:\n\n", answer2_without_knowledge.content)
print("\n")
time.sleep(2)

answer2_with_knowledge = retrieval_chain.invoke({"input": query2})

print("\nAnswer with knowledge:\n\n", answer2_with_knowledge['answer'])
print("\nContext Used:\n\n", answer2_with_knowledge['context'])
print("\n")
time.sleep(2)

pc.delete_index(index_name)

print("Completed!")

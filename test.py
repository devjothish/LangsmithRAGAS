from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.evaluation import evaluate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Load documents
loader = WebBaseLoader("https://en.wikipedia.org/wiki/New_York_City")
documents = loader.load()

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Initialize the model
model = ChatOpenAI(model="gpt-4")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Define question and get answer
question = "How did New York City get its name?"
result = qa_chain({"query": question})
print("\nAnswer:", result["result"])

# Prepare evaluation input
eval_input = {
    "question": question,
    "answer": result["result"],
    "contexts": [doc.page_content for doc in result["source_documents"]]
}

# Run evaluations
print("\nEvaluation Results:")
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
results = evaluate(eval_input, metrics=metrics)
print("\nDetailed Scores:")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
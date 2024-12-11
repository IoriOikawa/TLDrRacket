import os
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"

if "GOOGLE_API_KEY" not in os.environ:
    with open("./.env", "r") as mykey:        
        os.environ["GOOGLE_API_KEY"] = mykey.read().strip()

# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Embedding Function: Used when creating the DB, or making a query.
def get_embedding_function():
    # Bedrock Embeddings for AWS Deploy
    # embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    #)
    # Ollama Embeddings for Local Run
    # Install and 'ollama pull llama2|mistral' to deploy.
    # Use 'ollama serve' for restful API
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings

from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prepare the DB.
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
retriever = db.as_retriever(search_kwargs={"k" : 5})
# Generate query with the info augmented prompt.
# context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# prompt = prompt_template.format(context=context_text, question=query_text)
# print(prompt)

# model = Ollama(model="mistral")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
# response_text = model.invoke(prompt)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | model
    | StrOutputParser()
)

from flask import Flask, send_from_directory, request, jsonify

app = Flask(__name__,
            static_url_path='', 
            static_folder='static')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid request"}), 400

    user_message = data['message']
    response = rag_chain.invoke(user_message)
    
    contexts = db.similarity_search_with_score(user_message, k=5)
    sources = [doc.metadata.get("id", None) for doc, _score in contexts]
    
    return jsonify({"response": response, "sources": sources})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

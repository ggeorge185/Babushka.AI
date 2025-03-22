#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os

import argparse
import time
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = os.environ.get("MODEL", "Babushka")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

def initialize_model():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = Ollama(model=model)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

qa_model = initialize_model()

# Route to serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle questions from the front end
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")
    answer = qa_model(question)["result"]
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

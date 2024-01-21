from flask import Flask, render_template, request, jsonify
import json

# Assuming genai_sample_util and query_chat_model are defined in your_script.py
from your_script import genai_sample_util, populate_chunks_to_vector_db, query_chat_model

app = Flask(__name__)

# Initialize your system and vector database
genai_sample_util.print_sys_info()
client, collection = populate_chunks_to_vector_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    data = request.json
    query = data['query']
    print(f"\n\nQuery is: {query}\n")

    # Without RAG context
    system_role_content = "You are a helpful assistant"
    results_without_context = query_chat_model(system_role_content, query, "")

    # RAG enrichment
    system_role_content = "You are a helpful assistant. Use the following context to find the answer."
    results_with_context = collection.query(
        query_texts=[query],
        n_results=2
    )
    documents = results_with_context["documents"]
    results = query_chat_model(system_role_content, query, documents[0])

    # Send back the response to the front-end
    return jsonify({'answer': results["answer"]})

if __name__ == "__main__":
    app.run(debug=True)

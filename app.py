from flask import Flask, request, jsonify, render_template_string
import os
import sys
import logging

# Configure Flask logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = app.logger # Use Flask's logger

# Add the project root to the sys.path to allow importing rag.py
# This assumes app.py and rag.py are in the same directory.
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import the RAG system. This will initialize the RAGSystem class
# which in turn loads your fine-tuned model and FAISS index.
try:
    from rag import rag_system
except ImportError:
    logger.error("Could not import 'rag_system' from 'rag.py'. Please ensure rag.py exists and is correctly configured.")
    rag_system = None # Set to None to prevent further errors if import fails

# Define the HTML content for the chatbot interface directly in the Python file.
# In a larger project, this would typically be in a separate 'templates' folder.
CHATBOT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personalized Study Assistant Chatbot</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .chat-container {
            background-color: #ffffff;
            border-radius: 1.5rem; /* More rounded corners */
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            width: 90%;
            max-width: 768px; /* Max width for larger screens */
            display: flex;
            flex-direction: column;
            overflow: hidden;
            min-height: 600px;
            max-height: 90vh; /* Max height to fit screen */
        }
        .chat-header {
            background-color: #4f46e5; /* Indigo */
            color: white;
            padding: 1.5rem;
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
            border-top-left-radius: 1.5rem;
            border-top-right-radius: 1.5rem;
        }
        .chat-messages {
            flex-grow: 1;
            padding: 1.5rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message-bubble {
            max-width: 80%;
            padding: 0.75rem 1.25rem;
            border-radius: 1.25rem; /* Rounded bubbles */
            line-height: 1.4;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #e0e7ff; /* Light indigo */
            align-self: flex-end;
            border-bottom-right-radius: 0.25rem; /* Sharp corner for user */
        }
        .bot-message {
            background-color: #e2e8f0; /* Light gray */
            align-self: flex-start;
            border-bottom-left-radius: 0.25rem; /* Sharp corner for bot */
        }
        .chat-input-area {
            display: flex;
            padding: 1.5rem;
            border-top: 1px solid #e2e8f0;
            gap: 1rem;
        }
        .chat-input {
            flex-grow: 1;
            padding: 1rem;
            border: 1px solid #cbd5e1;
            border-radius: 0.75rem;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        .chat-input:focus {
            border-color: #6366f1; /* Darker indigo on focus */
        }
        .send-button {
            background-color: #4f46e5; /* Indigo */
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 0.75rem;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            border: none;
        }
        .send-button:hover {
            background-color: #4338ca; /* Darker indigo on hover */
            transform: translateY(-1px);
        }
        .send-button:active {
            transform: translateY(1px);
        }
        .loading-indicator {
            text-align: center;
            color: #6b7280;
            font-style: italic;
            margin-top: 1rem;
            display: none; /* Hidden by default */
        }
        .source-documents {
            background-color: #f0f2f5;
            padding: 0.75rem;
            border-radius: 0.75rem;
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: #4b5563;
            border: 1px dashed #cbd5e1;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            Personalized Study Assistant Chatbot
        </div>
        <div class="chat-messages" id="chat-messages">
            <div class="message-bubble bot-message">
                Hello! I'm your study assistant. Ask me anything about your course materials.
            </div>
        </div>
        <div class="loading-indicator" id="loading-indicator">
            Thinking...
        </div>
        <div class="chat-input-area">
            <input type="text" id="user-input" class="chat-input" placeholder="Ask a question...">
            <button id="send-button" class="send-button">Send</button>
        </div>
    </div>

    <script>
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');
        const chatMessages = document.getElementById('chat-messages');
        const loadingIndicator = document.getElementById('loading-indicator');

        function addMessage(message, sender, sourceDocs = []) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message-bubble');
            if (sender === 'user') {
                messageDiv.classList.add('user-message');
            } else {
                messageDiv.classList.add('bot-message');
            }
            messageDiv.textContent = message;

            chatMessages.appendChild(messageDiv);

            if (sourceDocs.length > 0) {
                const sourceDiv = document.createElement('div');
                sourceDiv.classList.add('source-documents');
                sourceDiv.innerHTML = '<strong>Sources:</strong><br>' + sourceDocs.map((doc, index) => `Doc ${index + 1}: ${doc.substring(0, 150)}...`).join('<br>');
                chatMessages.appendChild(sourceDiv);
            }

            chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom
        }

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        async function sendMessage() {
            const query = userInput.value.trim();
            if (query === '') return;

            addMessage(query, 'user');
            userInput.value = '';
            loadingIndicator.style.display = 'block'; // Show loading indicator
            sendButton.disabled = true; // Disable button during processing

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query }),
                });
                const data = await response.json();
                addMessage(data.response, 'bot', data.source_documents);
            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, I am having trouble connecting right now. Please try again later.', 'bot');
            } finally {
                loadingIndicator.style.display = 'none'; // Hide loading indicator
                sendButton.disabled = false; // Enable button
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Renders the chatbot's main HTML interface."""
    return render_template_string(CHATBOT_HTML)

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handles incoming chat queries, processes them using the RAG system,
    and returns the chatbot's response.
    """
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"response": "Please provide a question.", "source_documents": []}), 400

    logger.info(f"Received query from user: '{query}'")
    
    # Check if rag_system was initialized successfully
    if rag_system is None:
        logger.error("RAG system is not initialized. Cannot process query.")
        return jsonify({"response": "The chatbot backend is not ready. Please check server logs.", "source_documents": []}), 500

    # Use the globally initialized RAG system to get the response
    rag_response = rag_system.ask_question(query)
    
    logger.info(f"Sending response to user: '{rag_response['response']}'")
    return jsonify(rag_response)

if __name__ == '__main__':
    # Ensure the FAISS index and fine_tuned_model directories exist
    # and contain the necessary files before running the Flask app.
    if not os.path.exists("faiss_index"):
        logger.warning("FAISS index directory 'faiss_index' not found. Please run vectorize.py.")
    if not os.path.exists("fine_tuned_model"):
        logger.warning("Fine-tuned model directory 'fine_tuned_model' not found. Please run fine_tune.py.")

    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000) # Run on all available interfaces


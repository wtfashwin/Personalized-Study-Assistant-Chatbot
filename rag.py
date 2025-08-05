import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain.llms import HuggingFacePipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    A class to encapsulate the Retrieval-Augmented Generation (RAG) system.
    It loads the FAISS vector store and the fine-tuned QA model to answer questions.
    """
    def __init__(self, model_path: str = "./fine_tuned_model", faiss_index_path: str = "faiss_index"):
        """
        Initializes the RAG system by loading the necessary components.

        Args:
            model_path (str): Path to the directory containing the fine-tuned model and tokenizer.
            faiss_index_path (str): Path to the directory containing the FAISS index.
        """
        self.model_path = model_path
        self.faiss_index_path = faiss_index_path
        self.qa_chain = None
        self._initialize_rag_chain()

    def _initialize_rag_chain(self):
        """
        Loads the fine-tuned model, tokenizer, embeddings, FAISS index,
        and sets up the LangChain RetrievalQA chain.
        """
        logger.info("Loading tokenizer and fine-tuned model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading fine-tuned model or tokenizer from {self.model_path}: {e}")
            logger.error("Please ensure the fine_tune.py script ran successfully and saved the model.")
            return

        logger.info("Creating question-answering pipeline...")
        # Create a question-answering pipeline
        # device=-1 for CPU, or specify GPU device_id if available (e.g., device=0)
        # We force CPU usage for low-end device constraints.
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=-1, # Force CPU usage for low-end devices
            # max_length=512, # Adjust max_length for generated answer if needed
        )

        logger.info("Initializing HuggingFacePipeline for LangChain...")
        # Initialize HuggingFacePipeline with the QA pipeline
        llm = HuggingFacePipeline(pipeline=qa_pipeline)

        logger.info("Loading FAISS index...")
        try:
            # Re-initialize embeddings with the same model used during vectorization
            # This is crucial for consistent vector space.
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.load_local(self.faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading FAISS index from {self.faiss_index_path}: {e}")
            logger.error("Please ensure vectorize.py script ran successfully and created the index.")
            return

        logger.info("Creating RetrievalQA chain...")
        # Create a retrieval-based QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" combines all retrieved documents into one prompt
            retriever=db.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 relevant documents
            return_source_documents=True # Optionally return the source documents
        )
        logger.info("RAG chain initialized.")

    def ask_question(self, query: str) -> dict:
        """
        Asks a question using the RAG system and returns the response.

        Args:
            query (str): The question to ask.

        Returns:
            dict: A dictionary containing the answer and optionally source documents.
        """
        if self.qa_chain is None:
            logger.error("RAG chain not initialized. Cannot answer question.")
            return {"response": "Error: Chatbot not ready. Please check logs."}
        
        logger.info(f"Processing query: '{query}'")
        try:
            # The .invoke method is preferred over .run for newer LangChain versions
            # and provides a dictionary output.
            response = self.qa_chain.invoke({"query": query})
            # LangChain's RetrievalQA.invoke returns a dictionary with 'result' and 'source_documents'
            return {
                "response": response.get("result", "No answer found."),
                "source_documents": [doc.page_content for doc in response.get("source_documents", [])]
            }
        except Exception as e:
            # Log the full exception for better debugging
            logger.error(f"Error during RAG query: {e}", exc_info=True)
            return {"response": f"An error occurred while processing your request: {e}"}

# This part will be executed when rag.py is imported by app.py
# Initialize the RAG system globally so it's ready when Flask app starts
rag_system = RAGSystem()

if __name__ == '__main__':
    # Example usage if you run rag.py directly
    print("Running a test query from rag.py...")
    test_query = "What is the capital of France?"
    response = rag_system.ask_question(test_query)
    print(f"Question: {test_query}")
    print(f"Answer: {response['response']}")
    if response.get("source_documents"):
        print("\nSource Documents:")
        for i, doc in enumerate(response["source_documents"]):
            print(f"  Doc {i+1}: {doc[:200]}...") # Print first 200 chars of doc

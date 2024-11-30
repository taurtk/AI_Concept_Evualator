import os
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with your Groq API key
os.environ["GROQ_API_KEY"] = ""

# Initialize the Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize the Groq LLM
try:
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
    )
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None

# Define evaluation prompt template
evaluation_template = """
You are an AI evaluator. Your task is to assess the quality of an answer to a given question.

Question: {question}

Answer: {answer}

Please provide a comprehensive evaluation with the following structure:
**Total Score (out of 10): X**

**Strengths**
- Clear, specific strength point 1
- Clear, specific strength point 2
- Clear, specific strength point 3

**Areas for Improvement**
- Specific area of improvement 1
- Specific area of improvement 2
- Specific area of improvement 3

**Specific Suggestions for Enhancement**
- Detailed suggestion 1
- Detailed suggestion 2
- Detailed suggestion 3

Evaluation Breakdown:
1. Technical Accuracy: X/4 points
2. Depth of Understanding: X/3 points
3. Clarity and Coherence: X/2 points
4. Relevance to the Question: X/1 point
"""
evaluation_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=evaluation_template,
)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('submit_answer')
def handle_answer(data):
    logger.info(f"Received data: {data}")
    
    try:
        # Validate input
        if not data or 'question' not in data or 'answer' not in data:
            emit('feedback', {
                'evaluation': 'Invalid request. Please provide both question and answer.',
                'status': 'error'
            })
            return

        current_question = data['question']
        user_answer = data['answer']

        # Input validation
        if not current_question or not user_answer:
            emit('feedback', {
                'evaluation': 'Question and answer cannot be empty.',
                'status': 'error'
            })
            return

        # Check if LLM is initialized
        if llm is None:
            emit('feedback', {
                'evaluation': 'AI evaluation service is currently unavailable.',
                'status': 'error'
            })
            return

        # Generate prompt and get evaluation
        prompt = evaluation_prompt.format(question=current_question, answer=user_answer)
        
        logger.info(f"Generated prompt: {prompt}")
        
        evaluation = llm.predict(prompt)
        
        logger.info(f"Evaluation result:\n{evaluation}")

        # Emit feedback immediately after evaluation
        emit('feedback', {
            'evaluation': evaluation,
            'status': 'success'
        })
        
        logger.info(f"Evaluation completed for question: {current_question}")

    except Exception as e:
        logger.error(f"Unexpected error in handle_answer: {e}")
        emit('feedback', {
            'evaluation': 'An unexpected error occurred during evaluation.',
            'status': 'error'
        })

@socketio.on_error()
def error_handler(e):
    logger.error(f"SocketIO error: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
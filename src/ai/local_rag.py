import streamlit as st
# Set page config first
st.set_page_config(page_title="Teaching Simulation", page_icon="👩🏫")

import os
import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ------------------------------
# Set environment variables and initialize components
# ------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------
# Initialize session state
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "student_chat_history" not in st.session_state:
    st.session_state.student_chat_history = []

if "textbook_passages" not in st.session_state:
    st.session_state.textbook_passages = [
        "Effective classroom management involves clear expectations and consistency.",
        "Engaging lessons should include interactive activities and visual aids.",
        "Positive reinforcement can boost student motivation and participation.",
        "A reflective teacher reviews classroom interactions to improve teaching methods.",
        "Using varied instructional strategies can help meet diverse learning needs.",
    ]
    
    # Set up embeddings and FAISS index
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    passage_embeddings = embedder.encode(st.session_state.textbook_passages)
    passage_embeddings = np.array(passage_embeddings).astype('float32')
    
    dimension = passage_embeddings.shape[1]
    st.session_state.index = faiss.IndexFlatL2(dimension)
    st.session_state.index.add(passage_embeddings)
    st.session_state.embedder = embedder

# Initialize expert chat in session state if not present
if 'expert_chat_history' not in st.session_state:
    st.session_state.expert_chat_history = []

# ------------------------------
# Helper functions 
# ------------------------------
def check_appropriate_teacher_behavior(user_input):
    """Check if teacher's input is appropriate for a classroom with a 2nd grader"""
    try:
        # Create prompt to evaluate teacher behavior
        evaluation_prompt = {
            "role": "system",
            "content": """Evaluate whether the following teacher input is appropriate for a 2nd grade classroom.
            Inappropriate behavior includes:
            - Yelling or using ALL CAPS excessively
            - Using insulting or demeaning language
            - Making threats or using intimidation
            - Using inappropriate adult language or topics
            - Making personal comments unrelated to learning
            - Anything that would be considered verbal abuse
            
            Output ONLY "inappropriate" if the input is inappropriate, or "appropriate" if it is acceptable.
            Do not explain your reasoning - just return one word."""
        }
        
        # Send evaluation request to model
        response = ollama.chat(model="llama3.2", messages=[
            evaluation_prompt,
            {"role": "user", "content": f"Teacher's input: {user_input}"}
        ])
        result = response.get("message", {}).get("content", "").strip().lower()
        
        # Check if the model flagged it as inappropriate
        return "inappropriate" not in result
    except Exception as e:
        st.error(f"Error checking teacher behavior: {str(e)}")
        # Default to appropriate if the check fails
        return True

def get_response(user_input):
    """Function to get response from the Ollama model with chat history"""
    try:
        # First check if teacher behavior is appropriate
        is_appropriate = check_appropriate_teacher_behavior(user_input)
        
        if not is_appropriate:
            # If teacher behavior is inappropriate, don't respond
            return "[That reply was not appropriate.]"
        
        # Add teacher message to student chat history
        st.session_state.student_chat_history.append({"role": "user", "content": f"Teacher: {user_input}"})
        
        # System prompt for the student character
        system_prompt = {
            "role": "system", 
            "content": """You are an enthusiastic but sometimes distracted 2nd grade student in a classroom:
            - Use simple vocabulary that a 7-8 year old would know
            - Show curiosity and excitement about learning new things
            - Occasionally mention recess, lunch, or your friends
            - Keep responses short (2-3 sentences)
            - It's okay to be a little off-topic sometimes
            - Respond naturally to the teacher. Show appropriate emotions like excitement, confusion, or frustration."""
        }
        
        # Combine system prompt with chat history
        messages = [system_prompt] + st.session_state.student_chat_history
        
        # Get response from Ollama with full chat history
        response = ollama.chat(model="llama3.2", messages=messages)
        response_content = response.get("message", {}).get("content", "No response found.")
        
        # Add the student's response to chat history
        st.session_state.student_chat_history.append({"role": "assistant", "content": response_content})
        
        # Keep chat history at a reasonable size (last 10 exchanges)
        if len(st.session_state.student_chat_history) > 20:
            # Keep system message and last 19 messages
            st.session_state.student_chat_history = st.session_state.student_chat_history[-20:]
            
        return response_content
    except Exception as e:
        st.error(f"Error getting student response: {str(e)}")
        return "There was an issue with getting a response."

def retrieve_textbook_context(conversation_text, top_k=3):
    """Retrieve relevant textbook passages based on conversation context"""
    query_embedding = st.session_state.embedder.encode([conversation_text])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = st.session_state.index.search(query_embedding, top_k)
    retrieved_passages = [st.session_state.textbook_passages[i] for i in indices[0]]
    return retrieved_passages

def get_expert_advice(question, conversation_history):
    """Get advice from the expert teacher model with chat history"""
    try:
        # Format the conversation transcript
        conversation_transcript = "\n".join(
            f"{'Student' if msg['role'] == 'assistant' else 'Teacher'}: {msg['content']}" 
            for msg in conversation_history
        )
        
        # Get relevant teaching principles
        retrieved_passages = retrieve_textbook_context(conversation_transcript)
        passages_text = "\n".join(f"- {p}" for p in retrieved_passages)
        
        # Create or update expert chat history with system message
        if not st.session_state.expert_chat_history or st.session_state.expert_chat_history[0]["role"] != "system":
            system_message = {
                "role": "system",
                "content": "You are an expert teacher trainer providing concise, practical advice to a teacher interacting with a 2nd grade student. Focus on helpful strategies based on educational best practices."
            }
            st.session_state.expert_chat_history = [system_message]
        
        # Prepare the prompt with context
        user_message = {
            "role": "user",
            "content": f"""Question: {question}

Current conversation:
{conversation_transcript}

Teaching Principles to Consider:
{passages_text}"""
        }
        
        # Add user question to expert chat history
        st.session_state.expert_chat_history.append(user_message)
        
        # Keep chat history at a reasonable size
        if len(st.session_state.expert_chat_history) > 10:
            # Keep system message and last 9 exchanges
            st.session_state.expert_chat_history = [st.session_state.expert_chat_history[0]] + st.session_state.expert_chat_history[-9:]
        
        # Get response from Ollama with full expert chat history
        response = ollama.chat(model="llama3.2", messages=st.session_state.expert_chat_history)
        response_content = response.get("message", {}).get("content", "No response found.")
        
        # Add assistant response to expert chat history
        st.session_state.expert_chat_history.append({"role": "assistant", "content": response_content})
        
        return response_content
    except Exception as e:
        st.error(f"Error getting expert advice: {str(e)}")
        return "There was an issue getting expert advice."

# ------------------------------
# Streamlit UI
# ------------------------------
st.markdown("""
    <h1 style='font-size: 30px;'>Teacher-Student Interaction</h1>
    """, unsafe_allow_html=True)

# Display chat messages with labels
for message in st.session_state.messages:
    role = "Student" if message["role"] == "assistant" else "Teacher"
    with st.chat_message(message["role"]):
        st.markdown(f"**{role}:** {message['content']}")

# Chat input
if prompt := st.chat_input("Message the student..."):
    # Add teacher message to UI display history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"**Teacher:** {prompt}")
    
    # Get and display student response
    with st.chat_message("assistant"):
        response = get_response(prompt)
        st.markdown(f"**Student:** {response}")
    st.session_state.messages.append({"role": "assistant", "content": response})

# Create a sidebar for expert teacher consultation
with st.sidebar:
    st.markdown("""
        <h1 style='font-size: 24px;'>Expert Teacher Consultation</h1>
        <p style='font-size: 14px; color: #666;'>
            Ask an experienced teacher trainer for advice on the current situation.<br><br>
            <em>Example questions:</em>
            <ul>
                <li>How should I respond to this behavior?</li>
                <li>What strategy would work best here?</li>
                <li>How can I better engage this student?</li>
            </ul>
        </p>
        """, unsafe_allow_html=True)
    
    # Create a container for messages
    chat_container = st.container()
    
    # Create input container at the bottom
    input_container = st.container()
    
    # Place the input box first (at bottom due to reverse order)
    with input_container:
        expert_prompt = st.chat_input("Ask the expert teacher...", key="expert_chat_input")
    
    # Display messages in the chat container (excluding system message)
    with chat_container:
        for message in st.session_state.expert_chat_history:
            if message["role"] != "system":  # Skip system message in display
                role = "Expert" if message["role"] == "assistant" else "You"
                with st.chat_message(message["role"]):
                    st.markdown(f"**{role}:** {message['content']}")
    
    # Handle new messages
    if expert_prompt:
        with chat_container:
            # Add and display user message to UI
            with st.chat_message("user"):
                st.markdown(f"**You:** {expert_prompt}")
            
            # Get and display expert response
            with st.chat_message("assistant"):
                expert_response = get_expert_advice(expert_prompt, st.session_state.messages)
                st.markdown(f"**Expert:** {expert_response}")
    
    # Clear button for expert chat
    if len(st.session_state.expert_chat_history) > 1:  # Only show if there are messages beyond system
        if st.button("Clear Expert Chat History", type="secondary"):
            # Keep just the system message
            if st.session_state.expert_chat_history and st.session_state.expert_chat_history[0]["role"] == "system":
                st.session_state.expert_chat_history = [st.session_state.expert_chat_history[0]]
            else:
                st.session_state.expert_chat_history = []
            st.rerun()
    
    # Clear button for student chat
    if st.session_state.messages:
        if st.button("Clear Student Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.student_chat_history = []
            st.rerun()
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
def get_response(user_input):
    """Function to get response from the Ollama model"""
    try:
        # Enhanced prompt engineering for 2nd grade student simulation
        prompt = f"""Respond as an enthusiastic but sometimes distracted 2nd grade student in a classroom:
        - Use simple vocabulary that a 7-8 year old would know
        - Show curiosity and excitement about learning new things
        - Occasionally mention recess, lunch, or your friends
        - Keep responses short (2-3 sentences)
        - It's okay to be a little off-topic sometimes
        - Express emotions with phrases like "This is so cool!" or "I don't get it..."
        
        Student's response to: {user_input}"""
        
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        return response.get("message", {}).get("content", "No response found.")
    except Exception as e:
        print(f"Error getting Ollama response: {e}")
        return "There was an issue with getting a response."

def retrieve_textbook_context(conversation_text, top_k=3):
    query_embedding = st.session_state.embedder.encode([conversation_text])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = st.session_state.index.search(query_embedding, top_k)
    retrieved_passages = [st.session_state.textbook_passages[i] for i in indices[0]]
    return retrieved_passages

def get_expert_advice(question, conversation_history):
    try:
        conversation_transcript = "\n".join(
            f"{'Student' if msg['role'] == 'assistant' else 'Teacher'}: {msg['content']}" 
            for msg in conversation_history
        )
        
        retrieved_passages = retrieve_textbook_context(conversation_transcript)
        passages_text = "\n".join(f"- {p}" for p in retrieved_passages)
        
        prompt = f"""As an expert teacher, provide advice on this situation:

Question: {question}

Context:
{conversation_transcript}

Teaching Principles to Consider:
{passages_text}"""

        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        return response.get("message", {}).get("content", "No response found.")
    except Exception as e:
        print(f"Error getting expert advice: {e}")
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
    # Add teacher message
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
    
    # Display messages in the chat container
    with chat_container:
        for message in st.session_state.expert_chat_history:
            role = "Expert" if message["role"] == "assistant" else "You"
            with st.chat_message(message["role"]):
                st.markdown(f"**{role}:** {message['content']}")
    
    # Handle new messages
    if expert_prompt:
        with chat_container:
            # Add and display user message
            st.session_state.expert_chat_history.append({"role": "user", "content": expert_prompt})
            with st.chat_message("user"):
                st.markdown(f"**You:** {expert_prompt}")
            
            # Get and display expert response
            expert_response = get_expert_advice(expert_prompt, st.session_state.messages)
            st.session_state.expert_chat_history.append({"role": "assistant", "content": expert_response})
            with st.chat_message("assistant"):
                st.markdown(f"**Expert:** {expert_response}")
    
    # Clear button
    if st.session_state.expert_chat_history:
        if st.button("Clear Expert Chat History", type="secondary"):
            st.session_state.expert_chat_history = []
            st.rerun()
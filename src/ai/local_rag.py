#real working ollama code
 
import os
import streamlit as st
import ollama
 
# Set page config
st.set_page_config(page_title="Teacher-Student Interaction", page_icon="👩🏫")
 
# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
# ------------------------------
# Initialize session state
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# ------------------------------
# Helper function to interact with Ollama model
# ------------------------------
def get_response(user_input):
    """Function to get response from the Ollama model"""
    try:
        # Adding instruction to Ollama to respond as a 2nd grader
        prompt = f"Respond like a 2nd grader: {user_input}"
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        return response.get("message", {}).get("content", "No response found.")
    except Exception as e:
        print(f"Error getting Ollama response: {e}")
        return "There was an issue with getting a response."
 
# ------------------------------
# Streamlit UI
# ------------------------------
st.markdown("<h1 style='font-size: 30px;'>Teacher-Student Interaction</h1>", unsafe_allow_html=True)
 
# Display messages (Teacher-Student interaction)
for message in st.session_state.messages:
    role = "Student" if message["role"] == "assistant" else "Teacher"
    with st.chat_message(message["role"]):
        st.markdown(f"**{role}:** {message['content']}")
 
# Chat input for teacher-student interaction
if prompt := st.chat_input("Message the student..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"**Teacher:** {prompt}")
   
    # Get student response (as if it's a 2nd grader)
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f"**Student:** {response}")
 
# Sidebar for expert consultation (optional for teacher advice)
with st.sidebar:
    st.markdown("<h1 style='font-size: 24px;'>Expert Teacher Consultation</h1>", unsafe_allow_html=True)
    expert_prompt = st.chat_input("Ask the expert teacher...", key="expert_chat_input")
   
    # Display expert chat history
    if expert_prompt:
        st.session_state.messages.append({"role": "user", "content": expert_prompt})
        with st.chat_message("user"):
            st.markdown(f"**You:** {expert_prompt}")
       
        # Get expert advice (if needed)
        expert_response = get_response(expert_prompt)
        st.session_state.messages.append({"role": "assistant", "content": expert_response})
        with st.chat_message("assistant"):
            st.markdown(f"**Expert:** {expert_response}")
 
    # Clear chat history button
    if st.session_state.messages and st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.experimental_rerun()
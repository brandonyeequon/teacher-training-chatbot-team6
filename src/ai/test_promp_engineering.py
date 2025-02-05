import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ----------------------------------------------
# 1. Set up the Anthropic client and message history
# ----------------------------------------------
client = anthropic.Anthropic()
message_history = []  # This will store the conversation messages

# ----------------------------------------------
# 2. Set up the vector store with textbook passages
# ----------------------------------------------
# Example textbook passages that include best practices for teaching
textbook_passages = [
    "Effective classroom management involves clear expectations and consistency.",
    "Engaging lessons should include interactive activities.",
    "Positive reinforcement can boost student motivation and participation.",
    "A reflective teacher reviews classroom interactions to improve teaching methods.",
    "Using varied instructional strategies can help meet diverse learning needs.",
    # ... add additional passages as needed ...
]

# Load an embedding model (from SentenceTransformers)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for the textbook passages and convert to float32 numpy array
passage_embeddings = embedder.encode(textbook_passages)
passage_embeddings = np.array(passage_embeddings).astype('float32')

# Create a FAISS index for efficient similarity search
dimension = passage_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(passage_embeddings)

# ----------------------------------------------
# 3. Define helper functions
# ----------------------------------------------
def get_response(user_input):
    """
    Send the teacher's input to the simulated student and return the student's response.
    """
    # Record the teacher's message in the conversation history
    message_history.append({
        "role": "user",
        "content": [{"type": "text", "text": user_input}]
    })

    # Get a response from the simulated student
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        temperature=1,
        system=(
            "You are an 8-year-old student in a grumpy mood. "
            "You are speaking to an elementary school teacher. "
            "Give brief responses, unless you get agitated or excited."
            "Adapt your responces as the conversation progresses."
            "React to the teacher's tone and content."
            "If the problem is above a second grade level, give a incorrect anwser"
        ),
        messages=message_history
    )

    # Record the student's response in the conversation history
    message_history.append({
        "role": "assistant",
        "content": response.content
    })

    # Return the student's text (assuming the API returns a list of content objects)
    return response.content[0].text

def retrieve_textbook_context(conversation_text, top_k=3):
    """
    Retrieve the top_k textbook passages that are most relevant to the conversation.
    """
    # Compute the embedding for the conversation text
    query_embedding = embedder.encode([conversation_text])
    query_embedding = np.array(query_embedding).astype('float32')

    # Perform a similarity search in the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve and return the most similar textbook passages
    retrieved_passages = [textbook_passages[i] for i in indices[0]]
    return retrieved_passages

def generate_actionable_feedback(conversation_transcript):
    """
    Generate actionable feedback based on the conversation transcript and retrieved textbook passages.
    """
    # Retrieve relevant textbook passages using RAG
    retrieved_passages = retrieve_textbook_context(conversation_transcript)

    # Build a prompt that includes the conversation and textbook guidance
    prompt = f"""You are an experienced teacher trainer. Based on the following teacher conversation and relevant textbook guidance, provide actionable, constructive feedback on the teacher's performance.

Teacher Conversation:
---------------------
{conversation_transcript}

Relevant Textbook Guidance:
---------------------------
{"\n".join(retrieved_passages)}

Feedback:"""

    # Call the language model to generate feedback
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=1,
        system="You are an experienced teacher trainer who provides clear, actionable feedback.",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    )

    return response.content[0].text

def build_conversation_transcript():
    """
    Construct a transcript of the conversation from the message history.
    """
    transcript_lines = []
    for message in message_history:
        if message["role"] == "user":
            # Teacher's message (stored as a list of content dictionaries)
            teacher_text = message["content"][0]["text"]
            transcript_lines.append(f"Teacher: {teacher_text}")
        elif message["role"] == "assistant":
            # Student's message (assumed to be a list of content dictionaries)
            student_text = message["content"][0]["text"] if isinstance(message["content"], list) and "text" in message["content"][0] else str(message["content"])
            transcript_lines.append(f"Student: {student_text}")
    return "\n".join(transcript_lines)

# ----------------------------------------------
# 4. Main function: Chat and then generate feedback
# ----------------------------------------------
def main():
    print("Teacher-Student Chat (type 'quit' to finish conversation)")
    print("-" * 50)

    # Chat loop: teacher inputs messages until "quit" is entered
    while True:
        user_input = input("Teacher: ")
        if user_input.lower() == "quit":
            break

        # Get the student's response and display it
        student_response = get_response(user_input)
        print("Student:", student_response)

    # Build a transcript from the conversation history
    conversation_transcript = build_conversation_transcript()
    print("\nConversation Transcript:")
    print(conversation_transcript)

    # Generate and print actionable feedback based on the conversation
    print("\nGenerating actionable feedback...")
    feedback = generate_actionable_feedback(conversation_transcript)
    print("\nActionable Feedback:")
    print(feedback)

if __name__ == "__main__":
    main()




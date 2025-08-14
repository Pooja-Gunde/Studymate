import PyPDF2
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize Sentence Transformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📚 StudyMate ")
st.subheader("Simple PDF Q&A System")

# File upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Extract full text and split into chunks
full_text = ""
text_chunks = []
if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        full_text += page.extract_text() or ""
    
    chunk_size = 500
    text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Function to extract text under a specific heading
def extract_section(text, section_title):
    lines = text.split("\n")
    section_data = []
    capture = False
    for line in lines:
        if section_title.upper() in line.upper():  # Match heading
            capture = True
            continue
        if capture and line.isupper() and line.strip() != section_title.upper():
            break  # Stop at next heading
        if capture:
            section_data.append(line.strip())
    return "\n".join([l for l in section_data if l])

# Question input
question = st.text_input("Ask a question about the document:")

if question and text_chunks:
    # Find best matching chunk
    question_embedding = model.encode([question])
    chunk_embeddings = model.encode(text_chunks)
    similarities = cosine_similarity(question_embedding, chunk_embeddings)
    best_match_idx = similarities.argmax()
    context = text_chunks[best_match_idx]

    # Try extracting exact section if possible
    answer = extract_section(full_text, question)
    if not answer:  # Fallback if heading not found
        answer = context.strip()

    # Display only the final answer
    st.subheader("Answer:")
    st.write(answer)

st.caption("Now returns only the exact answer, without extra context or fluff.")

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AutoDoc AI", layout="wide")
st.title("📄 AutoDoc AI – Intelligent Document Q&A + Summarizer")

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Uploading..."):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        response = requests.post(f"{API_URL}/upload/", files=files)
        if response.status_code == 200:
            data = response.json()
            file_id = data["file_id"]
            st.session_state["file_id"] = file_id
            st.session_state["preview"] = data["preview"]
            st.success(f"Uploaded: {uploaded_file.name}")
        else:
            st.error("Upload failed. Please try again.")

# Show preview
if "preview" in st.session_state:
    st.subheader("📑 Document Preview")
    for i, page in enumerate(st.session_state["preview"]):
        st.markdown(f"**Page {i+1}:**")
        st.text_area(
            "Document text",
            value=page["text"],
            height=200,
            label_visibility="collapsed",
        )


# Ask question
if "file_id" in st.session_state:
    st.subheader("🤖 Ask a Question")
    question = st.text_input("Your question:")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            payload = {"file_id": st.session_state["file_id"], "question": question}
            res = requests.post(f"{API_URL}/query/", json=payload)
            if res.status_code == 200:
                st.markdown("**Answer:**")
                st.success(res.json()["answer"])
            else:
                st.error("Something went wrong!")

# Summarize
if "file_id" in st.session_state:
    st.subheader("📝 Summarize Document")
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            payload = {"file_id": st.session_state["file_id"]}
            res = requests.post(f"{API_URL}/summarize/", json=payload)
            if res.status_code == 200:
                st.markdown("**Summary:**")
                st.info(res.json()["summary"])
            else:
                st.error("Summarization failed.")

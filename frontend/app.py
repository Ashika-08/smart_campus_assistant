import streamlit as st
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Smart Campus Assistant", layout="wide")

# Session State Init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Sidebar
with st.sidebar:
    st.title("Campus Assistant")
    st.markdown("---")
    nav = st.radio("Navigate", ["Dashboard", "Chat Assistant", "Knowledge Graph", "Study Mode"])
    
    st.markdown("---")
    st.caption("Backend Status")
    try:
        requests.get(f"{API_URL}/docs", timeout=1)
        st.success("Online")
    except:
        st.error("Offline")

# --- DASHBOARD / UPLOAD ---
if nav == "Dashboard":
    st.header("📚 Dashboard & Upload")
    
    uploaded = st.file_uploader("Upload Course Material (PDF, DOCX, IMG)", type=["pdf", "docx", "png", "jpg", "jpeg"])
    
    if uploaded:
        if st.button("Process File"):
            with st.spinner("Uploading & Processing..."):
                files = {"file": (uploaded.name, uploaded, uploaded.type)}
                try:
                    resp = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                    if resp.status_code == 200:
                        st.success(f"Success! Processed {resp.json().get('chunks')} chunks.")
                        st.session_state.uploaded_file = uploaded.name
                        
                        # Trigger KG Build automatically
                        with st.spinner("Building Knowledge Graph..."):
                             requests.post(f"{API_URL}/build_kg", params={"filename": uploaded.name})
                             st.success("Knowledge Graph Built!")
                    else:
                        st.error(f"Failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.uploaded_file:
        st.info(f"Active File: **{st.session_state.uploaded_file}**")

# --- CHAT ASSISTANT ---
elif nav == "Chat Assistant":
    st.header("💬 Chat with your Materials")
    
    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    params = {"query": prompt, "hops": 1}
                    resp = requests.get(f"{API_URL}/hybrid_graph_answer", params=params, timeout=120)
                    if resp.status_code == 200:
                        ans = resp.json().get("answer", "No answer generated.")
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                    else:
                        st.error("Failed to get answer.")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- KNOWLEDGE GRAPH ---
elif nav == "Knowledge Graph":
    st.header("🕸️ Knowledge Graph")
    if not st.session_state.uploaded_file:
        st.warning("Please upload a file first.")
    else:
        if st.button("Refresh Graph Data"):
            try:
                # We don't have a full dump endpoint, but we can query specific entity or build again
                # For demo, let's just show stats if we stored them, or re-trigger build to get stats
                resp = requests.post(f"{API_URL}/build_kg", params={"filename": st.session_state.uploaded_file})
                if resp.status_code == 200:
                    data = resp.json()
                    col1, col2 = st.columns(2)
                    col1.metric("Nodes", data.get("num_nodes", 0))
                    col2.metric("Edges", data.get("num_edges", 0))
                    st.json(data.get("graph"))
            except Exception as e:
                st.error(f"Error: {e}")

# --- STUDY MODE ---
elif nav == "Study Mode":
    st.header("📝 Study Aids")
    
    tab1, tab2, tab3 = st.tabs(["Summary", "Quiz", "Flashcards"])
    
    with tab1:
        if st.button("Generate Summary"):
            if not st.session_state.uploaded_file:
                st.error("No file uploaded.")
            else:
                with st.spinner("Summarizing..."):
                    try:
                        resp = requests.get(f"{API_URL}/summary", params={"filename": st.session_state.uploaded_file}, timeout=300)
                        if resp.status_code == 200:
                            st.markdown(resp.json().get("summary"))
                        else:
                            st.error("Failed to generate summary.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
    with tab2:
        num_q = st.slider("Number of Questions", 1, 10, 3)
        if st.button("Generate Quiz"):
             if not st.session_state.uploaded_file:
                st.error("No file uploaded.")
             else:
                with st.spinner("Generating Quiz (this may take a moment)..."):
                    try:
                        resp = requests.get(f"{API_URL}/quiz", params={"filename": st.session_state.uploaded_file, "num_questions": num_q}, timeout=300)
                        if resp.status_code == 200:
                            st.session_state.quiz_data = resp.json().get("quiz", [])
                        else:
                            st.error("Failed to generate quiz.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if "quiz_data" in st.session_state:
            for i, q in enumerate(st.session_state.quiz_data):
                st.subheader(f"Q{i+1}: {q.get('question')}")
                opts = q.get('options', [])
                choice = st.radio(f"Select Answer for Q{i+1}", opts, key=f"q{i}")
                
                if st.button(f"Check Answer Q{i+1}", key=f"btn{i}"):
                    if choice == q.get('answer'):
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect. Correct answer: {q.get('answer')}")
                st.markdown("---")
                
    with tab3:
        if st.button("Generate Flashcards"):
            if not st.session_state.uploaded_file:
                st.error("No file uploaded.")
            else:
                with st.spinner("Creating Flashcards..."):
                    try:
                        resp = requests.get(f"{API_URL}/flashcards", params={"filename": st.session_state.uploaded_file, "num_cards": 5}, timeout=300)
                        if resp.status_code == 200:
                             st.session_state.flashcards = resp.json().get("flashcards", [])
                        else:
                             st.error("Failed.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if "flashcards" in st.session_state:
            for i, card in enumerate(st.session_state.flashcards):
                # Use expander to simulate flip
                with st.expander(f"Card {i+1}: {card.get('front')}"):
                    st.info(f"Answer: {card.get('back')}")

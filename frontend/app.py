import streamlit as st
import requests
import json
import io
from gtts import gTTS
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from streamlit_agraph import agraph, Node, Edge, Config


API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Smart Campus Assistant", layout="wide")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = None

def submit_chat():
    st.session_state.user_prompt = st.session_state.chat_input_box
    st.session_state.chat_input_box = ""


with st.sidebar:
    st.title("Campus Assistant")
    st.markdown("---")
    nav = st.radio("Navigate", ["Dashboard", "Chat Assistant", "Study Mode"])
    
    st.markdown("---")
    st.caption("Backend Status")
    try:
        requests.get(f"{API_URL}/docs", timeout=1)
        st.success("Online")
    except:
        st.error("Offline")


if nav == "Dashboard":
    st.header("📚 Dashboard & Upload")
    
   
    st.subheader("📤 Upload New Material")
    uploaded = st.file_uploader("Upload Course Material (PDF, DOCX, PPTX, IMG)", type=["pdf", "docx", "pptx", "png", "jpg", "jpeg"])
    
    if uploaded:
        if st.button("Process & Share File"):
            with st.spinner("Uploading to Repository..."):
                files = {"file": (uploaded.name, uploaded, uploaded.type)}
                try:
                    resp = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                    if resp.status_code == 200:
                        st.success(f"Success! Processed {resp.json().get('chunks')} chunks.")
                        st.session_state.uploaded_file = uploaded.name
                        
                        
                        with st.spinner("Building Knowledge Graph..."):
                             requests.post(f"{API_URL}/build_kg", params={"filename": uploaded.name})
                             st.success("Knowledge Graph Built!")
                        
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.uploaded_file:
        st.success(f"✅ Active Context: **{st.session_state.uploaded_file}**")
        
        
        if st.checkbox("🕸️ View Knowledge Graph", value=False):
            st.divider()
            st.subheader("Knowledge Graph")
            
            
            if st.button("Refresh Graph Data"):
                 
                 requests.post(f"{API_URL}/build_kg", params={"filename": st.session_state.uploaded_file})
            
            
            detail_level = st.select_slider(
                "Detail Level", 
                options=["Low", "Medium", "High", "All"], 
                value="Medium"
            )
            
            limit_map = {"Low": 20, "Medium": 50, "High": 100, "All": None}
            limit = limit_map[detail_level]
            
           
            with st.spinner(f"Visualizing ({detail_level})..."):
                try:
                    
                    resp = requests.get(f"{API_URL}/get_graph", params={"limit": limit})
                    if resp.status_code == 200:
                        data = resp.json()
                        kg_data = data.get("graph", {})
                        
                        if kg_data:
                            nodes = []
                            edges = []
                            for n in kg_data.get("nodes", []):
                                label_text = n.get("id", "?")
                                display_label = (label_text[:20] + '..') if len(label_text) > 20 else label_text
                                nodes.append(Node(
                                    id=n["id"], 
                                    label=display_label, 
                                    size=20,
                                    shape="dot",
                                    color="#FF4B4B",
                                    font={"color": "white"} 
                                ))
                            for e in kg_data.get("edges", []):
                                edges.append(Edge(
                                    source=e["source"], 
                                    target=e["target"], 
                                    label=e.get("relation", ""),
                                    color="#555"
                                ))
                                
                            config = Config(
                                width=900, 
                                height=600, 
                                directed=True, 
                                physics=True, 
                                hierarchical=False,
                                
                                barnesHut={
                                    "gravitationalConstant": -20000,
                                    "centralGravity": 0.3, 
                                    "springLength": 120, 
                                    "springConstant": 0.04, 
                                    "damping": 0.09, 
                                    "avoidOverlap": 1
                                }
                            )
                            st.write(f"**Nodes:** {len(nodes)} | **Edges:** {len(edges)}")
                            agraph(nodes=nodes, edges=edges, config=config)
                        else:
                            st.info("No relationships found.")
                except Exception as e:
                    st.error(f"Graph Error: {e}")
            st.divider()



# --- CHAT ASSISTANT ---
# --- CHAT ASSISTANT ---
elif nav == "Chat Assistant":
    st.header("💬 Chat with your Materials")
    
    
    with st.container():
        c1, c2 = st.columns([12, 1]) 
        
        with c1:
            user_text = st.text_input(
                "Ask a question...",
                key="chat_input_box",
                placeholder="Type your question...",
                label_visibility="collapsed",
                on_change=submit_chat
            )
            
        with c2:
            mic_clicked = st.button("🎤", key="mic_btn", help="Speak", use_container_width=True)

    st.divider()

    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])



    voice_text = None

    if mic_clicked:
        r = sr.Recognizer()
        with st.spinner("Listening..."):
            try:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source)
                    audio_data = r.listen(source, timeout=5)
                voice_text = r.recognize_google(audio_data)
                st.session_state.last_voice_input = voice_text
            except Exception as e:
                st.error(f"Mic Error: {e}")

    if "last_voice_input" in st.session_state:
        voice_text = st.session_state.last_voice_input
        del st.session_state.last_voice_input

    
    prompt = voice_text or st.session_state.user_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    params = {"query": prompt, "hops": 1}
                    resp = requests.get(f"{API_URL}/hybrid_graph_answer",
                                        params=params, timeout=120)
                    
                    if resp.status_code == 200:
                        ans = resp.json().get("answer", "No answer generated.")
                        st.markdown(ans)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": ans}
                        )

                       
                        try:
                            audio_stream = io.BytesIO()
                            tts = gTTS(text=ans, lang='en')
                            tts.write_to_fp(audio_stream)
                            st.audio(audio_stream, format='audio/mp3', autoplay=False)
                        except Exception as e:
                            st.warning(f"TTS Error: {e}")

                    else:
                        st.error("Failed to get answer from backend.")

                except Exception as e:
                    st.error(f"Error: {e}")
        
        
        st.session_state.user_prompt = None


elif nav == "Study Mode":
    st.header("📝 Study Aids")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Quiz", "Flashcards", "Study Planner 📅"])
    
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
                
                with st.expander(f"Card {i+1}: {card.get('front')}"):
                    st.info(f"Answer: {card.get('back')}")

    with tab4:
        st.header("Smart Study Planner 📅")
        days = st.slider("Plan Duration (Days)", 3, 14, 7)
        if st.button("Generate Study Plan"):
            if not st.session_state.uploaded_file:
                st.error("No file uploaded.")
            else:
                with st.spinner("Planning your schedule..."):
                    try:
                        resp = requests.get(f"{API_URL}/plan", params={"filename": st.session_state.uploaded_file, "days": days}, timeout=300)
                        if resp.status_code == 200:
                            plan = resp.json().get("plan", [])
                            st.success(f"Generated {len(plan)}-Day Plan!")
                            for day in plan:
                                with st.expander(f"Day {day['day']}: {day['topic']}", expanded=True):
                                    for activity in day['activities']:
                                        st.write(f"- {activity}")
                        else:
                            st.error("Failed to generate plan.")
                    except Exception as e:
                        st.error(f"Error: {e}")

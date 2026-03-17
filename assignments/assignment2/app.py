import streamlit as st
import json

st.set_page_config(page_title="LLM Debate Pipeline Visualizer", layout="wide", page_icon="⚖️")

# CSS to make the UI pop
st.markdown("""
<style>
    .judge-panel {
        background-color: #1E1E2E;
        border-left: 5px solid #F9A826;
        padding: 20px;
        border-radius: 5px;
        color: #F8F8F2;
    }
    .confidence {
        font-size: 1.2rem;
        font-weight: bold;
        color: #50FA7B;
    }
    .question-title {
        color: #FF79C6;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    try:
        with open("data/results_log.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Log file not found. Ensure the debate pipeline has run successfully.")
        return []
    except json.JSONDecodeError:
        st.error("Log file is corrupted or empty.")
        return []

def main():
    st.sidebar.title("⚖️ Agentic Debate Viewer")
    st.sidebar.write("Explore full transcripts and judge findings from the multi-agent LLM debate pipeline.")
    
    data = load_data()
    
    if not data:
        return

    # Sidebar question selector
    questions = [entry["question"] for entry in data]
    selected_idx = st.sidebar.selectbox("Select a Debate Question:", range(len(questions)), format_func=lambda i: questions[i])
    
    # Get selected debate record
    debate = data[selected_idx]
    
    # Top Information
    st.markdown(f"## <span class='question-title'>Question:</span> {debate['question']}", unsafe_allow_html=True)
    st.markdown(f"**Ground Truth:** {'True' if debate['ground_truth'] else 'False'}")
    
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗣️ Debate Transcript")
        
        # Render the chat log
        for msg in debate["transcript"]:
            role = msg["role"]
            content = msg["content"]
            
            # Use distinct avatars for visual separation
            avatar = "🛡️" if role == "Proponent" else "🗡️"
            
            with st.chat_message(role, avatar=avatar):
                st.markdown(f"**{role} (Round {msg['round']})**")
                st.markdown(content)
                
    with col2:
        st.subheader("🧑‍⚖️ Judge's Verdict")
        
        judge_output = debate.get("judge_output", "No judgement rendered.")
        confidence_score = debate.get("confidence_score")
        extracted_verdict = debate.get("extracted_verdict")
        
        st.markdown("<div class='judge-panel'>", unsafe_allow_html=True)
        st.markdown(judge_output)
        
        if extracted_verdict is not None:
            st.markdown(f"**Extracted Verdict:** {extracted_verdict}", unsafe_allow_html=True)

        if confidence_score is not None:
            st.markdown(f"<p class='confidence'>Confidence Score: {confidence_score} / 5</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

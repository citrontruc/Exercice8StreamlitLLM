import streamlit as st
from helper.conversation_agent import ConversationAgent
from helper.ui_helper import UIHelper

#Sidebar
st.sidebar.markdown("Chat avec GPT 🤖")
st.sidebar.markdown("Page de chat qui permet d'échanger avec le modèle GPT-35.")

#Page principale
st.title("Chat avec GPT 🤖")

pdf_uiagent = UIHelper("🤖")
competency_analysis_agent = ConversationAgent()

# The messages between user and assistant are kept in the session_state (the local storage)
if "message_hist" not in st.session_state:
    st.session_state.message_hist = []

# Message de bienvenue à l'utilisateur et bouton "nouveau chat"
pdf_uiagent.reset_button()

# Affichage de l'historique des messages
if st.session_state.message_hist == []:
    pdf_uiagent.initialize_conv(competency_analysis_agent.random_intro())
else:
    pdf_uiagent.show_conversation(st.session_state.message_hist)

# This is the user's textbox for chatting with the assistant
if prompt := st.chat_input("Quelle est votre question ?"):
    # Lors de la réception d'un message, on affiche le message, on récupère la réponse et on l'affiche à l'écran.
    pdf_uiagent.format_user_question(prompt)
    streamed_response = competency_analysis_agent.get_answer_llm_async(
            message_hist=st.session_state.message_hist[-2:], 
            user_question = prompt
        )
    full_str_reponse = pdf_uiagent.format_llm_response(streamed_response)

    # On met à jour notre historique de messages
    st.session_state.message_hist.append({"role": "user", "content": prompt})
    st.session_state.message_hist.append({"role": "assistant", "content": full_str_reponse})
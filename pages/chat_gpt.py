import streamlit as st
from helper.conversation_agent import ConversationAgent

#Sidebar
st.sidebar.markdown("Chat avec GPT 🤖")
st.sidebar.markdown("Page de chat qui permet d'échanger avec le modèle GPT-35.")

#Page principale
st.title("Chat avec GPT 🤖")

left, midleft, mid, rightmid, right = st.columns(5)
competency_analysis_agent = ConversationAgent()
avatars = {"user": "😃", "assistant": "🤖"}

# Ajout d'informations utilisateur et construction d'un chat.
st.write("Que puis-je faire pour vous aider ?")
left, right = st.columns(2)
with left:
    if st.button("Nouveau chat"):
        st.session_state.message_hist = []

# The messages between user and assistant are kept in the session_state (the local storage)
if "message_hist" not in st.session_state:
    st.session_state.message_hist = []

# Nous envoyons un message d'introduction à l'utilisateur afin d'améliorer l'expérience utilisateur.
if st.session_state.message_hist == []:
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        message_placeholder = st.empty()
        bot_message = ""
        for streamed_content in competency_analysis_agent.random_intro():
            message_placeholder.markdown(streamed_content + "▌")
        message_placeholder.markdown(streamed_content)
        st.session_state.message_hist.append({"role" : "assistant", "content" : streamed_content})

else:
    # Display chat messages from history on app rerun
    for message in st.session_state.message_hist:
        with st.chat_message(message["role"], avatar=avatars[message["role"]]):
            st.markdown(message["content"])


# This is the user's textbox for chatting with the assistant
if prompt := st.chat_input("Quelle est votre question ?"):
    generation_finished = False
    st.write("Un instant s'il vous plait ⌛...")

    # When the user sends a message...
    new_message = {"role": "user", "content": prompt}
    st.session_state.message_hist.append(new_message)
    
    with st.chat_message("user", avatar=avatars["user"]):
        st.markdown(prompt)

    # ... the assistant replies
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        message_placeholder = st.empty()
        full_str_response = ""
        # We ask our conversation_agent to answer
        streamed_response = competency_analysis_agent.get_answer_llm_async(
            message_hist=st.session_state.message_hist[-2:], 
            user_question = prompt
        )
        for resp in streamed_response:
            full_str_response += resp or ""
            message_placeholder.markdown(full_str_response + "▌")
        message_placeholder.markdown(full_str_response)
        generation_finished = True
    
    st.session_state.message_hist.append(
        {"role": "assistant", "content": full_str_response}
    )
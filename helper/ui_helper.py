"""
Contient un ensemble de fonctions liées à streamlit afin de prendre en charge l'UI des pages streamlit.

"""

import streamlit as st

class UIHelper():
    def __init__(self, avatar) -> None:
        self.AVATAR = avatar
        self.AVATAR_DICT = {"user": "😃", "assistant": self.AVATAR}

    def create_layout(self):
        left, midleft, mid, rightmid, right = st.columns(5)
        avatars = self.AVATAR_DICT
    
    def greet(self):
        # Ajout d'informations utilisateur et construction d'un chat.
        st.write("Que puis-je faire pour vous aider ?")
        left, right = st.columns(2)
        with left:
            if st.button("Nouveau chat"):
                st.session_state.message_hist = []
    
    def initialize_conv(self, text_intro):
        with st.chat_message("assistant", avatar=self.AVATAR):
            message_placeholder = st.empty()
            for streamed_content in text_intro:
                message_placeholder.markdown(streamed_content + "▌")
            message_placeholder.markdown(streamed_content)
            st.session_state.message_hist.append({"role" : "assistant", "content" : streamed_content})
    
    def show_conversation(self, message_history):
        for message in message_history:
            with st.chat_message(message["role"], avatar=self.AVATAR_DICT[message["role"]]):
                st.markdown(message["content"])
    
    def format_user_question(self, prompt):
        st.write("Un instant s'il vous plait ⌛...")
        with st.chat_message("user", avatar=self.AVATAR_DICT["user"]):
            st.markdown(prompt)

    def format_llm_response(self, streamed_response):
        with st.chat_message("assistant", avatar=self.AVATAR_DICT["assistant"]):
            message_placeholder = st.empty()
            full_str_response = ""
            # We ask our conversation_agent to answer
            for resp in streamed_response:
                full_str_response += resp or ""
                message_placeholder.markdown(full_str_response + "▌")
            message_placeholder.markdown(full_str_response)
        return full_str_response
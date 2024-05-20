"""
Contient un ensemble de fonctions liées à streamlit afin de prendre en charge l'UI des pages streamlit.

"""

import streamlit as st

class UIHelper():
    """
    Classe pour uniformiser le design des différentes pages de chat.
    Gère l'affichage des messages et l'UI de façon générale.
    """

    def __init__(self, avatar) -> None:
        self.AVATAR = avatar
        self.AVATAR_DICT = {"user": "😃", "assistant": self.AVATAR}
    
    def reset_button(self):
        """
        Création du bouton nouveau chat afin de réinitialiser l'historique de conversation.
        """
        left, right = st.columns(2)
        with left:
            if st.button("Nouveau chat"):
                st.session_state.message_hist = []
    
    def initialize_conv(self, text_intro):
        """
        Si on ne possède pas de texte dans l'historique, affiche une formule d'accueil (text_intro).

        input:
            text_intro (str)
        """
        with st.chat_message("assistant", avatar=self.AVATAR):
            message_placeholder = st.empty()
            for streamed_content in text_intro:
                message_placeholder.markdown(streamed_content + "▌")
            message_placeholder.markdown(streamed_content)
            st.session_state.message_hist.append({"role" : "assistant", "content" : streamed_content})
    
    def show_conversation(self, message_history):
        """
        Si on ne possède un historique, affiche notre historique de text (message_history).

        input:
            message_history (list)
        """
        for message in message_history:
            with st.chat_message(message["role"], avatar=self.AVATAR_DICT[message["role"]]):
                st.markdown(message["content"])
    
    def format_user_question(self, prompt):
        """
        Affiche dans le chat la question qui vient d'être posée.

        input:
            prompt (str)
        """
        with st.chat_message("user", avatar=self.AVATAR_DICT["user"]):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=self.AVATAR_DICT["assistant"]):
            st.markdown("Un instant s'il vous plait ⌛...")

    def format_llm_response(self, streamed_response):
        """
        Prend en entrée un générateur de message asynchrone, récupère le texte du générateur au fur et à mesure qu'il vient et l'affiche.

        input:
            streamed_response (generator)
        output:
            full_str_response (str)
        """
        with st.chat_message("assistant", avatar=self.AVATAR_DICT["assistant"]):
            message_placeholder = st.empty()
            full_str_response = ""
            # We ask our conversation_agent to answer
            for resp in streamed_response:
                full_str_response += resp or ""
                message_placeholder.markdown(full_str_response + "▌")
            message_placeholder.markdown(full_str_response)
        return full_str_response
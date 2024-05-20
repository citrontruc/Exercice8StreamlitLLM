"""
Contient un ensemble de fonctions afin de pouvoir créer un agent conversationnel avec lequel l'utilisateur peut discuter.

"""

from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai._streaming import Stream
import os
import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Dict, Generator, Any, Optional

load_dotenv()

OPENAI_KEY = os.environ.get("OPENAI_KEY")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_KEY)
with open("data/rag_prompt.txt") as file:
    RAG_PROMPT_TEMPLATE = "".join(file.readlines())
SYSTEM_PROMPT = "Commence tes phrases par 'shiver me timbers matelot !'"
RAG_DB = pd.read_csv("data/text_rag.csv", sep=";")
RAG_VECTOR = np.load("data/vector_rag.npy")
RAG_DB["embedding"] = list(RAG_VECTOR)

# Le chargement d'un modèle d'embedding est long. Si la partie RAG ne vous intéresse pas, supprimer cette ligne.
EMBEDDING_MODEL = SentenceTransformer("distiluse-base-multilingual-cased")

class ConversationAgent:
    """
    Crée un agent conversationnel qui gère les interactions avec l'utilisateur.
    Pour l'aider, il peut appeler de l'IA générative.
    """
    
    def __init__(self) -> None:
        self.OPENAI_KEY = OPENAI_KEY
        self.OPENAI_CLIENT = OPENAI_CLIENT
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.RAG_PROMPT_TEMPLATE = RAG_PROMPT_TEMPLATE
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.RAG_DB = RAG_DB

    def random_intro(self) -> Generator[str, Any, None]:
        """
        Envoie un message de bienvenue à l'utilisateur.
        Permet de clarifier le fonctionnement du bot. Il faut que l'utilisateur sache ce qu'il peut faire.
        """
        chosen_intro = random.choice(
            [
                "Bonjour les humains, comment est-ce que je peux vous aider ?",
                "Posez-moi toutes vos questions, j'essaierai d'y répondre au mieux !",
                "PKDNDABDOANLKS?OAHNFONADPAJ?DP ! Oh, désolé j'étais ailleurs."
            ]
        )
        # Create a streaming effect
        splitted_text = chosen_intro.split(" ")
        for i, word in enumerate(splitted_text):
            yield " ".join(splitted_text[: i + 1])
            time.sleep(0.05)


    # Une fonction pour mettre en forme la question utilisateur et l'inclure à un prompt plus complet et plus complexe
    def format_user_question(self, user_question, prompt_template, documentation):
        """
        Fonction qui prend en entrée une question utilisateur pour renvoyer une question modifiée.

        input :
            user_question (str)

        output :
            promptAnswer (str)
        """
        promptAnswer = prompt_template.replace("{{query}}", user_question)
        promptAnswer = promptAnswer.replace("{{documentation}}", ".\nDocument suivant : ".join(documentation))
        return promptAnswer


    def ask_llm(self, user_question, message_hist, temperature = 0.5, top_p = 0.5, prompt_template="{{query}}", documentation=[]):
        """
        Dans cette fonction, nous allons récupérer une demande utilisateur et essayer d'y répondre.
        Nous allons trouver les documents en rapport avec la demande pour finalement demander à un LLM de trouver une réponse à la question.

        input :
            user_question (str)
            message_hist (list)
            model_name (str)
            temperature (float)
            top_p (float)

        output :
            promptAnswer (str)
        """
        formatted_question = self.format_user_question(user_question=user_question, prompt_template=prompt_template, documentation=documentation)

        # Construction d'une réponse
        print(formatted_question)
        gpt_response = self.OPENAI_CLIENT.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + message_hist +
                        [{"role": "user", "content": formatted_question}],
            temperature = temperature,
            max_tokens = 512,
            top_p = top_p,
            stream=True)
        
        for response in gpt_response:
            if len(response.choices) > 0:
                yield response.choices[0].delta.content
            else:
                yield ""
            time.sleep(0.05)


    def get_answer_llm_async(
        self,
        message_hist: List[Dict[str, str]],
        user_question: str,
        temperature=0.5,
        top_p=0.5,
        prompt_template="{{query}}", 
        documentation=[]
    ) -> Generator[Optional[str], Any, None]:
        """
        Méthode asynchrone qui renvoie une réponse de LLM à l'utilisateur sous forme de stream.
        
        input :
            message_hist (dict)
            user_question (str)

        output :
            un générateur qui permet de récupérer les réponses de l'API LLM.
        """
        # Nous récupérons les réponses en stream afin de pouvoir les afficher dès réception.
        streaming_response: Stream[
            ChatCompletionChunk
        ] = self.ask_llm(user_question=user_question, message_hist=message_hist, temperature=temperature, top_p=top_p, prompt_template=prompt_template, documentation=documentation)
        full_response = ""
        # When you iterate on the stream, you get a token for every response
        for response in streaming_response:
            yield response
            if response:
                full_response += response
            time.sleep(0.05)

    def search_doc(
            self,
            user_question: str,
            num_docs: int=5
    ):
        """
        Méthode qui à une question posée retourne les morceaux de textes les plus pertinents dans la documentation.
        
        input :
            user_question (str)
            num_docs (int)

        output :
            chunk_texts (array)
        """
        embedded_question = self.EMBEDDING_MODEL.encode(user_question)
        similarity = self.RAG_DB["embedding"].apply(lambda x : cosine_similarity([embedded_question], [x])[0][0])
        closest_chunk_indexes = similarity.sort_values().tail(num_docs).index
        chunk_texts = self.RAG_DB.iloc[closest_chunk_indexes]["text_chunk"].values
        return chunk_texts
    
    def answer_rag(
        self,
        message_hist: List[Dict[str, str]],
        user_question: str,
        temperature=0.5,
        top_p=0.5,
        num_docs=5
    ) -> Generator[Optional[str], Any, None]:
        documentation = self.search_doc(user_question, num_docs=num_docs)
        return self.get_answer_llm_async(message_hist=message_hist, user_question=user_question, temperature=temperature, top_p=top_p, prompt_template=self.RAG_PROMPT_TEMPLATE, documentation=documentation)
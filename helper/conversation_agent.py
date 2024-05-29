"""
Contient un ensemble de fonctions afin de pouvoir créer un agent conversationnel avec lequel l'utilisateur peut discuter.

"""

import cohere
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai._streaming import Stream
import os
import pandas as pd
from PyPDF2 import PdfReader
import random
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Dict, Generator, Any, Optional

load_dotenv()

OPENAI_KEY = os.environ.get("OPENAI_KEY")
COHERE_KEY = os.environ.get("COHERE_KEY")
COHERE_CLIENT = cohere.Client(COHERE_KEY)
OPENAI_CLIENT = OpenAI(api_key=OPENAI_KEY)
COHERE_MODEL = "rerank-multilingual-v3.0"
SYSTEM_PROMPT = "Commence tes phrases par 'shiver me timbers matelot !'"
with open("data/rag_prompt.txt") as file:
    RAG_PROMPT_TEMPLATE = "".join(file.readlines())
LOCAL_RAG_DB = pd.read_csv("data/text_rag.csv", sep=";")
RAG_VECTOR = np.load("data/vector_rag.npy")
LOCAL_RAG_DB["embedding"] = list(RAG_VECTOR)
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased"

class ConversationAgent:
    """
    Crée un agent conversationnel qui gère les interactions avec l'utilisateur.
    Pour l'aider, il peut appeler de l'IA générative.
    """
    
    def __init__(self, RAG_DOC=False) -> None:
        self.OPENAI_KEY = OPENAI_KEY
        self.OPENAI_CLIENT = OPENAI_CLIENT
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.COHERE_KEY = COHERE_KEY
        self.COHERE_CLIENT = COHERE_CLIENT
        self.COHERE_MODEL = COHERE_MODEL

        # Etapes longues. Ne pas faire systématiquement.
        if RAG_DOC:
            self.RAG_PROMPT_TEMPLATE = RAG_PROMPT_TEMPLATE
            self.EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    def set_rag_source(self, source="local"):
        """
        Quand un utilisateur veut utiliser le RAG, initialise la source à interroger.
        input:
            source (pdf ou "local")
        """
        if source=="local":
            self.rag_source = LOCAL_RAG_DB
        else:
            self.rag_source = self.prepare_for_rag(source)


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
            num_docs: int=10
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
        similarity = self.rag_source["embedding"].apply(lambda x : cosine_similarity([embedded_question], [x])[0][0])
        closest_chunk_indexes = similarity.sort_values().tail(num_docs).index
        chunk_texts = self.rag_source.iloc[closest_chunk_indexes]["text_chunk"].values
        return chunk_texts
    
    def rerank_doc(
            self,
            user_question: str,
            documents: list,
            top_n: int = 3
    ):
        """
        Nous avons récupérer les documents utilisateurs pertinents. Nous voulons les classés afin de ne garder que ceux qui répondent à la question utilisateur.
        
        input :
            user_question (str)
            documents (list)
            top_n (int)

        output :
            new_documentation (list)
    
    """
        response = self.COHERE_CLIENT.rerank(
            model=self.COHERE_MODEL,
            query=user_question,
            documents=documents.tolist(),
            top_n=top_n
        )
        new_documentation = []
        for element in response.results:
            new_documentation.append(documents[element.index])
        return new_documentation
    
    def answer_rag(
        self,
        message_hist: List[Dict[str, str]],
        user_question: str,
        temperature=0.5,
        top_p=0.5,
        num_docs=10
    ) -> Generator[Optional[str], Any, None]:
        documentation = self.search_doc(user_question, num_docs=num_docs)
        documentation = self.rerank_doc(user_question, documents=documentation)
        return self.get_answer_llm_async(message_hist=message_hist, user_question=user_question, temperature=temperature, top_p=top_p, prompt_template=self.RAG_PROMPT_TEMPLATE, documentation=documentation)

    def prepare_for_rag(self, pdf_document):
        """
        Méthode qui prend en entrée un pdf et le traite afin de pouvoir en faire un dataframe contenant des chunks et leur embedding.
        input:
            pdf_document
        output:
            embedding_dataframe (pandas dataframe)
        """
        document_text = self.extract_text(pdf_document)
        sentence_list = self.split_in_sentences(document_text)
        chunk_list = self.separate_in_chunks(sentence_list)
        embedding_dataframe = self.create_dataframe_from_chunks(chunk_list)
        return embedding_dataframe

    def extract_text(self, pdf_document):
        """
        Extrait le texte de toutes les pages d'un pdf. Ne récupère pas les figures, les schemas ou les images.
        input:
            pdf_document
        output:
            full_pdf_text (str)
        """
        reader = PdfReader(pdf_document)
        parts = []
        for i in range(len(reader.pages)):
            parts.append(reader.pages[i].extract_text())
        full_pdf_text = " ".join(parts)
        return full_pdf_text

    def split_in_sentences(self, text_str):
        """
        Prend un texte et le sépare en phrase en utilisant la ponctuation.
        input:
            text_str (str)
        output:
            sentence_list (list)
        """
        sentence_list = re.split('\\. |\\! |\\? ', text_str)
        for i in range(len(sentence_list)):
            sentence_list[i] = sentence_list[i].replace("\n", " ")
        return sentence_list
    
    def separate_in_chunks(self, sentence_list):
        """
        Prend une liste de phrase et les sépare en chunks. ATTENTION : ignore les phrases "trop longues" pour l'instant (désolé Proust).
        input:
            sentence_list (str)
        output:
            chunk_list (list)
        """
        chunk_list = []
        current_chunk = ""
        threshold = 1024
        i=0

        sentence_list = [element for element in sentence_list if len(element) < 400]
        while i < len(sentence_list):
            accept_new_sentence = (len(current_chunk) + len(sentence_list[i])) < threshold
            if accept_new_sentence:
                current_chunk = current_chunk + ". " + sentence_list[i]
            else:
                chunk_list.append(current_chunk)
                current_chunk = ""
                if i > 0:
                    current_chunk += sentence_list[i-1] + ". " + sentence_list[i]
                else:
                    current_chunk += sentence_list[i]
            i+=1

        chunk_list.append(current_chunk)
        return chunk_list
    
    def create_dataframe_from_chunks(self, chunk_list):
        """
        Récupère une liste de chunks et en fait l'embedding afin de créer un dataframe python qui pourra ensuite être utilisé pour faire du RAG.
        input:
            chunk_list (list)
        output:
            embedding_dataframe (pandas dataframe)
        """
        embedding_data = self.EMBEDDING_MODEL.encode(chunk_list)
        embedding_dataframe = pd.DataFrame({"text_chunk" : chunk_list, "embedding" : list(embedding_data)})
        return embedding_dataframe
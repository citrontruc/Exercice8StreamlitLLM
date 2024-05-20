contact : clement.lionceau@gmail.com
date : 20/05/2024

---

# Application LLM en streamlit
---

## But du repository

Ce repository contient plusieurs scripts qui constituent une application streamlit de lecture de documents. L'application permet notamment d'interroger un modèle de LLM (GPT), d'interroger un document dans le cadre de rag et finalement d'interroger un pdf utilisateur.

---

## Contenu 

Le repository est constitué de trois dossiers et d'un fichier main.py.
- main.py : fonction principale de notre streamlit.
- pages : contient les différentes pages de notre application streamlit. Chaque page représente un onglet de l'application.
- data : dossier contenant les données utilisées dans le RAG ainsi que le notebook qui a servi à les mettre en forme.
- helper : contient des scripts utilisées par les pages de l'application streamlit.

---

## Exécuter le code

Afin d'exécuter ce notebook, vous aurez besoin d'un compte openai et d'une clé d'api OPENAI. Appelez cette clé OPENAI_KEY et enregistrez la dans un fichier .env à la racine du projet.
Une fois que cela est fait, exécutez la commande "streamlit run main.py" à la racine du projet.

---
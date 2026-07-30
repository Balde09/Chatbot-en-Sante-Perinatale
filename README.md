# Chatbot en santé périnatale

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)](https://www.gradio.app/)
[![Chroma](https://img.shields.io/badge/Chroma-0.5+-purple.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black.svg)](https://ollama.com/)
[![python-dotenv](https://img.shields.io/badge/python--dotenv-1.x-lightgrey.svg)](https://pypi.org/project/python-dotenv/)
[![Unstructured](https://img.shields.io/badge/Unstructured-0.15+-red.svg)](https://unstructured.io/)
[![lxml](https://img.shields.io/badge/lxml-5.x-blueviolet.svg)](https://lxml.de/)

</div>

Assistant conversationnel bilingue pour la périnatalité, propulsé par une recherche augmentée de type RAG et une interface Gradio élégante.

Le projet utilise :
- des documents XML dans `data_xml/`
- une base vectorielle Chroma locale dans `chroma_db_bilingual/`
- des embeddings Ollama avec `nomic-embed-text`
- un modèle de chat Ollama avec `llama3.1`
- une interface web Gradio pour discuter en français ou en anglais

## Fonctionnalités

- Réponses bilingues FR / EN
- Détection simple de la langue et gestion des messages mixtes
- Contexte conversationnel court grâce à un cache de questions-réponses
- Demandes de précision automatiques quand une question manque de contexte
- Deux scripts d’ingestion XML selon le niveau de vitesse et de détail voulu

## Prérequis

- Python 3.10 ou plus récent
- [Ollama](https://ollama.com)
- Les modèles Ollama suivants installés localement :
  - `nomic-embed-text`
  - `llama3.1`

Exemple :

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure du projet

- `chatbot.py` : lance l’assistant Gradio et interroge la base Chroma
- `ingest_xml_database.py` : ingestion XML avec chargement via `UnstructuredXMLLoader`
- `ingest_xml_fast.py` : ingestion plus rapide basée sur `xml.etree.ElementTree`
- `data_xml/` : corpus XML source
- `chroma_db_bilingual/` : base vectorielle persistée utilisée par le chatbot

## Ingestion des documents

Si vous voulez reconstruire la base vectorielle, lancez l’un des deux scripts suivants :

```bash
python ingest_xml_fast.py
```

ou, pour une ingestion plus structurée :

```bash
python ingest_xml_database.py
```

Les deux scripts écrivent dans `chroma_db_bilingual/` avec la collection `perinatalite_bilingual`.

## Lancement du chatbot

Une fois la base vectorielle prête, démarrez l’application :

```bash
python chatbot.py
```

L’interface Gradio s’ouvre dans le navigateur. Le script démarre avec `share=True`, donc Gradio peut aussi générer un lien de partage public si nécessaire.

## Notes

- Le projet est centré sur l’information périnatale et ne remplace pas un avis médical.
- L’assistant invite l’utilisateur à contacter un professionnel de santé en cas de besoin urgent.
- Si vous ajoutez de nouveaux XML, il faut relancer l’ingestion pour mettre à jour la base Chroma.

#!/usr/bin/env python3
"""
Script d'ingestion rapide des fichiers XML
Utilise xml.etree.ElementTree pour un parsing plus rapide
Chunks de 2000 caractères pour préserver le contexte médical
"""

from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

DATA_PATH = "data_xml"
CHROMA_PATH = "chroma_db_bilingual"
COLLECTION = "perinatalite_bilingual"

print("🚀 Début de l'ingestion rapide des fichiers XML...")

# 1) Embeddings locaux Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2) Trouver tous les fichiers XML
xml_files = list(Path(DATA_PATH).rglob("*.xml"))
print(f"📁 Trouvé {len(xml_files)} fichiers XML")

# 3) Charger et parser les XML avec ElementTree (beaucoup plus rapide)
documents = []
for i, xml_file in enumerate(xml_files, 1):
    try:
        # Parser le XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extraire tout le texte
        text = ET.tostring(root, encoding='unicode', method='text')
        
        # Créer un document
        doc = Document(
            page_content=text,
            metadata={"source": str(xml_file)}
        )
        documents.append(doc)
        
        # Afficher la progression tous les 500 fichiers
        if i % 500 == 0:
            print(f"  Chargé {i}/{len(xml_files)} fichiers...")
            
    except Exception as e:
        print(f"  ⚠️ Erreur lors du chargement de {xml_file}: {e}")
        continue

print(f"✅ {len(documents)} documents chargés")

# 4) Découpage en chunks - AUGMENTÉ pour garder le contexte médical complet
print("✂️ Découpage en chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,      # Chunks plus grands pour préserver le contexte
    chunk_overlap=200     # Overlap pour éviter de perdre l'information
)
chunks = splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunks créés")

# 5) Nettoyer les métadonnées
for chunk in chunks:
    # Garder seulement la source
    source = chunk.metadata.get("source", "unknown.xml")
    chunk.metadata = {"source": str(source)}

# 6) Créer le vectorstore
print("💾 Création du vectorstore...")
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# 7) Ajouter les documents par lots
batch_size = 1000
total_chunks = len(chunks)

print(f"📤 Ajout de {total_chunks} chunks par lots de {batch_size}...")

for i in range(0, total_chunks, batch_size):
    batch_chunks = chunks[i:i + batch_size]
    batch_ids = [str(uuid4()) for _ in range(len(batch_chunks))]
    
    batch_num = i // batch_size + 1
    total_batches = (total_chunks + batch_size - 1) // batch_size
    
    print(f"Lot {batch_num}/{total_batches} ({len(batch_chunks)} documents)")
    vector_store.add_documents(batch_chunks, ids=batch_ids)

print(f"✅ TERMINÉ ! {len(chunks)} chunks ingérés depuis {len(documents)} fichiers XML.")
print(f"📊 Collection: {COLLECTION}")
print(f"💾 Répertoire: {CHROMA_PATH}")

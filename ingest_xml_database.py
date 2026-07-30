from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import DirectoryLoader, UnstructuredXMLLoader

load_dotenv()

DATA_PATH = "data_xml"
CHROMA_PATH = "chroma_db_bilingual"  # Nouvelle base pour FR + EN
COLLECTION = "perinatalite_bilingual"

# 1) Embeddings locaux Ollama (gratuit, illimité)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2) Charger tous les .xml du dossier (et sous-dossiers)
loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.xml",
    loader_cls=UnstructuredXMLLoader,
    loader_kwargs={"mode": "elements"}  # segmentation logique par balises
)
raw_docs = loader.load()

# 3) Découpage en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(raw_docs)

# Nettoyer les métadonnées complexes manuellement
def clean_metadata(metadata):
    """Nettoie les métadonnées en gardant seulement les types simples"""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        elif isinstance(value, list) and len(value) > 0:
            # Convertir les listes en string
            cleaned[key] = str(value[0]) if value else ""
        else:
            # Convertir tout le reste en string
            cleaned[key] = str(value)
    return cleaned

for d in chunks:
    # Nettoyer les métadonnées
    d.metadata = clean_metadata(d.metadata)
    # Normaliser la source
    src = d.metadata.get("source") or d.metadata.get("filename") or "unknown.xml"
    d.metadata["source"] = str(src)

# 4) Persist Chroma
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

ids = [str(uuid4()) for _ in range(len(chunks))]
if chunks:
    # Traiter par lots de 1000 documents max pour éviter l'erreur de batch trop gros
    batch_size = 1000
    total_chunks = len(chunks)
    
    print(f"Traitement de {total_chunks} chunks par lots de {batch_size}...")
    
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        print(f"Traitement du lot {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch_chunks)} documents)")
        vector_store.add_documents(batch_chunks, ids=batch_ids)

print(f"✅ Ingesté {len(chunks)} chunks depuis {len(raw_docs)} fichiers XML.")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from collections import deque
import hashlib

load_dotenv()

CHROMA_PATH = "chroma_db_bilingual"
COLLECTION = "perinatalite_bilingual"

class QACache:
    def __init__(self, max_size=10):
        self.cache = deque(maxlen=max_size)
        self.query_cache = {}
    
    def _hash_query(self, query):
        """Crée un hash de la question pour la recherche rapide"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query):
        """Récupère une réponse du cache si elle existe"""
        query_hash = self._hash_query(query)
        return self.query_cache.get(query_hash)
    
    def add(self, query, answer):
        """Ajoute une Q&R au cache"""
        query_hash = self._hash_query(query)
        qa_pair = {"query": query, "answer": answer, "hash": query_hash}
        
        if query_hash in self.query_cache:
            self.cache = deque([item for item in self.cache if item["hash"] != query_hash], maxlen=10)
        
        self.cache.append(qa_pair)
        self.query_cache[query_hash] = answer
        
        if len(self.cache) == 10 and len(self.query_cache) > 10:
            current_hashes = {item["hash"] for item in self.cache}
            self.query_cache = {h: ans for h, ans in self.query_cache.items() if h in current_hashes}
    
    def get_recent_context(self):
        """Retourne le contexte des conversations récentes pour le LLM"""
        if not self.cache:
            return ""
        
        context = "\n\n--- Conversations récentes ---\n"
        for i, qa in enumerate(list(self.cache)[-5:], 1):
            context += f"Q{i}: {qa['query']}\nR{i}: {qa['answer'][:200]}...\n\n"
        return context

# Initialiser le cache global
qa_cache = QACache(max_size=10)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

llm = ChatOllama(
    model="llama3.1",
    temperature=0.3,
    num_ctx=2048,
    num_predict=512
)

vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def detect_language(text):
    """Détecte la langue du texte (FR, EN, ou MIXED)"""
    # Mots-clés français EXCLUSIFS (n'existent pas en anglais)
    french_only = ["le", "la", "les", "un", "une", "des", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
                   "est", "sont", "et", "de", "du", "dans", "sur", "pour", "avec", "que", "qui", "quoi",
                   "comment", "quand", "où", "pourquoi", "bonjour", "merci", "oui", "non", "quel", "quelle",
                   "mon", "ma", "mes", "son", "sa", "ses", "ce", "cette", "ces", "au", "aux"]
    
    # Mots-clés anglais EXCLUSIFS (n'existent pas en français)
    english_only = ["the", "is", "are", "that", "what", "how", "when", "where", "why", 
                    "hello", "hi", "thank", "thanks", "yes", "can", "could", "would", "should",
                    "my", "your", "his", "her", "our", "their", "this", "these", "those"]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # Compter uniquement les mots EXCLUSIFS à chaque langue
    french_count = sum(1 for word in words if word in french_only)
    english_count = sum(1 for word in words if word in english_only)
    
    # Détection des caractères accentués français (indicateur très fort)
    has_french_accents = any(char in text_lower for char in ['à', 'â', 'é', 'è', 'ê', 'ë', 'ç', 'ù', 'û', 'ï', 'î', 'ô'])
    
    # Si accents français, c'est forcément français (sauf si mélange évident)
    if has_french_accents:
        if english_count > 2:  # Plus de 2 mots anglais exclusifs = mélange
            return "mixed"
        return "fr"
    
    # Détecter un vrai mélange (les deux langues présentes de manière significative)
    if french_count >= 2 and english_count >= 2:
        return "mixed"
    
    # Décision basée sur le compte
    if french_count > english_count:
        return "fr"
    elif english_count > french_count:
        return "en"
    
    # Par défaut français (au Québec)
    return "fr"

def respond(message, history):
    lang = detect_language(message)
    
    if lang == "mixed":
        print("Langues mélangées détectées")
        return "Je vous prie de poser votre question dans une seule langue, soit en français, soit en anglais. Merci de votre compréhension. 😊\n\nPlease ask your question in one language only, either French or English. Thank you for your understanding. 😊"
    
    print(f"Langue détectée: {'Français' if lang == 'fr' else 'English'}")
    
    greetings = ["bonjour", "bonsoir", "salut", "hello", "hi", "coucou", "hey"]
    farewell = ["au revoir", "bye", "merci", "à bientôt", "goodbye", "adieu"]
    question_words = ["comment", "quoi", "que", "qui", "où", "quand", "pourquoi", "est-ce que", "?"]
    
    message_lower = message.lower().strip()
    
    is_short_message = len(message.split()) <= 3
    
    # Mots-clés étendus pour détection des détails (FR + EN)
    detail_keywords = [
        # Français
        "détail", "expliqu", "développ", "approfondi", "précis", "complet",
        "savoir", "information", "comment", "pourquoi", "mécanisme", "processus",
        "cause", "symptôme", "traitement", "conseil", "davantage", "suite",
        # Anglais
        "detail", "explain", "elaborate", "more", "expand", "comprehensive",
        "know", "information", "how", "why", "mechanism", "process",
        "cause", "symptom", "treatment", "advice", "further"
    ]
    
    # Vérifier si le message contient des indicateurs de demande de détails
    contains_detail_keywords = any(keyword in message_lower for keyword in detail_keywords)
    
    # Détecter si c'est probablement une demande de détails (FR + EN)
    is_likely_detail_request = (
        is_short_message and contains_detail_keywords
    ) or any(phrase in message_lower for phrase in [
        # Français
        "plus de", "en savoir", "expliquer", "développer", "approfondir",
        "comment ça", "pourquoi", "détails", "précisions", "compléments",
        # Anglais
        "more about", "tell me more", "explain", "elaborate", "expand on",
        "how does", "why", "details", "clarify", "more info"
    ])
    
    # Si c'est probablement une demande de détails ET qu'il y a une conversation précédente
    if is_likely_detail_request and qa_cache.cache:
        last_conversation = list(qa_cache.cache)[-1]
        previous_question = last_conversation["query"]
        print(f"Demande de détails détectée pour: {previous_question[:50]}...")
        
        # IMPORTANT: Détecter la langue de la QUESTION PRÉCÉDENTE pour maintenir la cohérence
        previous_lang = detect_language(previous_question)
        print(f"Langue de la question originale: {'Français' if previous_lang == 'fr' else 'English'}")
        print(f"Langue de la demande de détails: {'Français' if lang == 'fr' else 'English'}")
        print(f"Réponse sera en: {'Français' if previous_lang == 'fr' else 'English'}")
        
        # Utiliser la question précédente pour la recherche
        docs = retriever.invoke(previous_question)
        if not docs:
            if previous_lang == "fr":
                answer = f"Je n'ai pas trouvé d'informations supplémentaires pour votre question précédente : '{previous_question}'"
            else:
                answer = f"I couldn't find additional information for your previous question: '{previous_question}'"
            qa_cache.add(message, answer)
            return answer
        knowledge = "\n\n".join(d.page_content for d in docs)
        sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")][:3]
        recent_context = qa_cache.get_recent_context()
        
        # Prompt spécial pour les détails (adapté selon la langue de la QUESTION ORIGINALE)
        if previous_lang == "fr":
            prompt = f"""Tu es un assistant expert en PÉRINATALITÉ spécialisé dans l'accompagnement médical et psychologique pendant la grossesse, l'accouchement et la période postnatale.

INSTRUCTIONS:
- Réponds UNIQUEMENT en français
- Base tes réponses EXCLUSIVEMENT sur les connaissances fournies ci-dessous
- Adopte un ton professionnel, empathique et rassurant
- Fournis une réponse COMPLÈTE et DÉTAILLÉE sur la question précédente
- Explique les mécanismes, causes, symptômes, traitements, précautions
- Inclus des exemples concrets et des conseils pratiques
- Réponds de façon NATURELLE et CONVERSATIONNELLE, SANS titres, sans sections, sans markdown
- Ne structure PAS avec des titres comme "Causes:", "Symptômes:", "Traitements:"
- Écris en paragraphes fluides et naturels comme une conversation
- Si l'information n'existe pas dans les connaissances, dis: "Je ne trouve pas d'informations supplémentaires dans ma base de connaissances."

{recent_context}

QUESTION PRÉCÉDENTE À DÉTAILLER: {previous_question}
DEMANDE DE DÉTAILS: {message}

CONNAISSANCES SPÉCIALISÉES EN PÉRINATALITÉ:
{knowledge}

RÉPONSE DÉTAILLÉE:"""
        else:  # English
            prompt = f"""You are an expert PERINATAL assistant specializing in medical and psychological support during pregnancy, childbirth, and the postnatal period.

INSTRUCTIONS:
- Respond ONLY in English
- Base your answers EXCLUSIVELY on the knowledge provided below
- Adopt a professional, empathetic, and reassuring tone
- Provide a COMPLETE and DETAILED answer about the previous question
- Explain mechanisms, causes, symptoms, treatments, precautions
- Include concrete examples and practical advice
- Respond in a NATURAL and CONVERSATIONAL manner, WITHOUT titles, sections, or markdown
- Do NOT structure with titles like "Causes:", "Symptoms:", "Treatments:"
- Write in fluid, natural paragraphs like a conversation
- If the information doesn't exist in the knowledge, say: "I cannot find additional information in my knowledge base."

{recent_context}

PREVIOUS QUESTION TO DETAIL: {previous_question}
DETAIL REQUEST: {message}

SPECIALIZED PERINATAL KNOWLEDGE:
{knowledge}

DETAILED ANSWER:"""
        
        answer = llm.invoke(prompt).content
        
        # Supprimer les sources - plus de références
        qa_cache.add(message, answer)
        print(f"💾 Réponse détaillée ajoutée au cache pour question précédente.")
        return answer
    
    # Initialiser les variables pour la gestion des clarifications
    context_enrichment = ""
    skip_clarification_check = False
    
    # Détecter si c'est une réponse à une question de clarification
    if qa_cache.cache:
        last_conversation = list(qa_cache.cache)[-1]
        last_answer = last_conversation.get("answer", "")
        last_question = last_conversation.get("query", "")
        
        # Vérifier si la dernière réponse était une demande de clarification
        is_clarification_request = any(phrase in last_answer.lower() for phrase in [
            "quel âge", "depuis combien de temps", "quelle semaine de grossesse",
            "how old", "how long", "how many weeks"
        ])
        
        # Vérifier si le message actuel est court (probablement une réponse à la clarification)
        is_short_answer = len(message.split()) <= 10
        
        if is_clarification_request and is_short_answer:
            print(f"🔄 Réponse à clarification détectée: {message}")
            print(f"📝 Question originale: {last_question}")
            print(f"❓ Clarification demandée: {last_answer}")
            
            # Vérification simple: si réponse est juste un nombre sans unité pour l'âge, demander précision
            import re
            numbers = re.findall(r'\d+', message)
            if len(numbers) == 1 and message.strip().isdigit():
                if any(phrase in last_answer.lower() for phrase in ["quel âge", "how old"]):
                    if lang == "fr":
                        return f"{message.strip()} semaines, {message.strip()} mois ou {message.strip()} ans ?"
                    else:
                        return f"{message.strip()} weeks, {message.strip()} months or {message.strip()} years?"
            
            # Enrichir le contexte avec l'information de clarification
            if any(phrase in last_answer.lower() for phrase in ["quel âge", "how old"]):
                # C'est une réponse d'âge
                if lang == "fr":
                    if "an" in message.lower() or "année" in message.lower():
                        context_enrichment = f"\n[CONTEXTE IMPORTANT: enfant déjà né, âgé de {message}]"
                    else:
                        context_enrichment = f"\n[CONTEXTE IMPORTANT: bébé déjà né, âgé de {message}, PAS une grossesse]"
                else:
                    context_enrichment = f"\n[IMPORTANT CONTEXT: baby already born, {message} old, NOT pregnancy]"
                print(f"Contexte enrichi avec âge: {message}")
            elif any(phrase in last_answer.lower() for phrase in ["depuis combien", "how long"]):
                # C'est une réponse de durée
                if lang == "fr":
                    context_enrichment = f"\n[symptômes depuis {message}]"
                else:
                    context_enrichment = f"\n[symptoms for {message}]"
                print(f"Contexte enrichi avec durée: {message}")
            
            # Utiliser la question originale pour la recherche
            message = last_question
            skip_clarification_check = True
    
    # Vérifier le cache normal pour les nouvelles questions uniquement
    if not skip_clarification_check:
        cached_answer = qa_cache.get(message)
        if cached_answer:
            print(f"🔄 Réponse récupérée du cache pour: {message[:50]}...")
            return cached_answer
    
    # Vérifier si c'est une salutation simple (sans question médicale)
    is_greeting = any(greet in message_lower for greet in greetings)
    is_farewell = any(fare in message_lower for fare in farewell)
    has_question = any(qword in message_lower for qword in question_words) or len(message.split()) > 3
    
    if (is_greeting or is_farewell) and not has_question:
        if is_greeting:
            if lang == "fr":
                answer = """Bonjour et bienvenue ! 

Je suis votre assistant spécialisé en **périnatalité**. Je peux vous aider avec des questions sur :

**La grossesse** - suivi, symptômes, développement
**L'accouchement** - préparation, déroulement, récupération  
**La période postnatale** - allaitement, soins du nouveau-né
**Le soutien psychologique** - bien-être émotionnel

N'hésitez pas à me poser vos questions !"""
            else:  # English
                answer = """Hello and welcome! 

I'm your specialized **perinatal** assistant. I can help you with questions about:

**Pregnancy** - monitoring, symptoms, development
**Childbirth** - preparation, process, recovery  
**Postnatal period** - breastfeeding, newborn care
**Psychological support** - emotional well-being

Feel free to ask me your questions!"""
        else:
            if lang == "fr":
                answer = """Merci pour votre visite ! 

J'espère avoir pu vous aider dans vos questions sur la périnatalité. 

N'hésitez pas à revenir si vous avez d'autres interrogations. Prenez soin de vous ! """
            else:  # English
                answer = """Thank you for your visit! 

I hope I was able to help you with your questions about perinatal care. 

Feel free to come back if you have any other questions. Take care! """
        
        qa_cache.add(message, answer)
        print(f"Salutation traitée: {message[:30]}...")
        return answer
    
    # Récupérer les documents pertinents (seulement pour les nouvelles questions)
    docs = retriever.invoke(message)
    
    if not docs:
        if lang == "fr":
            answer = "Je n'ai pas trouvé d'informations pertinentes dans ma base de connaissances pour répondre à votre question sur la périnatalité."
        else:
            answer = "I couldn't find relevant information in my knowledge base to answer your question about perinatal care."
        qa_cache.add(message, answer)
        return answer
    
    knowledge = "\n\n".join(d.page_content for d in docs)
    sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")][:3]

    # Récupérer le contexte des conversations récentes
    recent_context = qa_cache.get_recent_context()

    # Détecter si la question manque de détails et nécessite des clarifications (sauf si on vient de traiter une clarification)
    needs_clarification = False
    clarification_questions = []
    
    if not skip_clarification_check:
        # Analyser la question pour identifier les informations manquantes
        message_lower_check = message.lower()
        
        # Vérifier si la question concerne un enfant/bébé sans mention d'âge
        if any(word in message_lower_check for word in ["enfant", "bébé", "bebe", "nourrisson", "petit", "child", "baby", "infant"]):
            if not any(word in message_lower_check for word in ["mois", "semaine", "an", "année", "jour", "month", "week", "year", "old", "âge", "age"]):
                needs_clarification = True
                if lang == "fr":
                    clarification_questions.append("Quel âge a votre enfant ?")
                else:
                    clarification_questions.append("How old is your child?")
        
        # Vérifier si la question concerne des symptômes sans durée
        if any(word in message_lower_check for word in ["symptôme", "douleur", "mal", "fièvre", "vomissement", "saignement", "symptom", "pain", "fever", "vomit", "bleeding"]):
            if not any(word in message_lower_check for word in ["depuis", "jour", "semaine", "heure", "matin", "hier", "for", "since", "days", "hours", "yesterday"]):
                needs_clarification = True
                if lang == "fr":
                    clarification_questions.append("Depuis combien de temps avez-vous ces symptômes ?")
                else:
                    clarification_questions.append("How long have you had these symptoms?")
        
        # Vérifier si la question concerne la grossesse sans mention du trimestre/semaine
        if any(word in message_lower_check for word in ["enceinte", "grossesse", "pregnant", "pregnancy"]) and "?" in message:
            if not any(word in message_lower_check for word in ["trimestre", "semaine", "mois", "trimester", "week", "month", "combien", "how many", "how far"]):
                needs_clarification = True
                if lang == "fr":
                    clarification_questions.append("À quelle semaine de grossesse êtes-vous ?")
                else:
                    clarification_questions.append("How many weeks pregnant are you?")
    
    # Si des clarifications sont nécessaires, les demander UNE À LA FOIS
    if needs_clarification and clarification_questions:
        first_question = clarification_questions[0]
        print(f"❓ Clarification nécessaire: {first_question}")
        # CRUCIAL: Mettre la clarification dans le cache pour pouvoir la détecter plus tard
        qa_cache.add(message, first_question)
        return first_question

    # Pour les nouvelles questions (pas de demande de détails), toujours donner un résumé
    if lang == "fr":
        response_type = "RÉSUMÉ"
        detail_instruction = """- Fournis un RÉSUMÉ COURT et ESSENTIEL (2-4 phrases maximum)
- Va à l'essentiel avec les points clés
- Réponds de façon NATURELLE et CONVERSATIONNELLE, SANS titres, sans markdown"""
    else:
        response_type = "SUMMARY"
        detail_instruction = """- Provide a SHORT and ESSENTIAL SUMMARY (2-4 sentences maximum)
- Get straight to the key points
- Respond in a NATURAL and CONVERSATIONAL manner, WITHOUT titles, without markdown"""
    print(f"📋 Résumé court généré pour nouvelle question: {message[:50]}...")

    # Prompt optimisé pour la périnatalité avec contexte des conversations (adapté selon la langue)
    if lang == "fr":
        prompt = f"""Tu es un assistant expert en PÉRINATALITÉ spécialisé dans l'accompagnement médical et psychologique pendant la grossesse, l'accouchement et la période postnatale.

INSTRUCTIONS:
- Réponds UNIQUEMENT en français
- Base tes réponses EXCLUSIVEMENT sur les connaissances fournies ci-dessous
- Adopte un ton professionnel, empathique et rassurant
- Pour les QUESTIONS MÉDICALES: Va DIRECTEMENT au contenu, SANS formules de politesse
- Ne commence PAS les réponses médicales par "Bonjour", "Je suis ravi", etc.
- Réponds de façon NATURELLE, SANS titres en gras, SANS sections marquées
- Ne structure PAS avec des titres comme "**Causes:**", "**Symptômes:**"
- Écris en paragraphes fluides comme une conversation naturelle
- Si l'information n'existe pas dans les connaissances, dis: "Je ne trouve pas cette information dans ma base de connaissances sur la périnatalité."
- Privilégie les informations les plus récentes et basées sur des preuves scientifiques
- En cas de question médicale urgente, rappelle de consulter un professionnel de santé
- Prends en compte le contexte des conversations récentes si pertinent

TYPE DE RÉPONSE: {response_type}
{detail_instruction}

{recent_context}

QUESTION ACTUELLE: {message}{context_enrichment}

CONNAISSANCES SPÉCIALISÉES EN PÉRINATALITÉ:
{knowledge}

RÉPONSE EXPERTE:"""
    else:  # English
        prompt = f"""You are an expert PERINATAL assistant specializing in medical and psychological support during pregnancy, childbirth, and the postnatal period.

INSTRUCTIONS:
- Respond ONLY in English
- Base your answers EXCLUSIVELY on the knowledge provided below
- Adopt a professional, empathetic, and reassuring tone
- For MEDICAL QUESTIONS: Go DIRECTLY to the content, WITHOUT polite formulas
- Do NOT start medical answers with "Hello", "I'm delighted", etc.
- Respond in a NATURAL manner, WITHOUT bold titles, WITHOUT marked sections
- Do NOT structure with titles like "**Causes:**", "**Symptoms:**"
- Write in fluid paragraphs like a natural conversation
- If the information doesn't exist in the knowledge, say: "I cannot find this information in my perinatal care knowledge base."
- Prioritize the most recent information based on scientific evidence
- For urgent medical questions, remind to consult a healthcare professional
- Take into account the context of recent conversations if relevant

RESPONSE TYPE: {response_type}
{detail_instruction}

{recent_context}

CURRENT QUESTION: {message}{context_enrichment}

SPECIALIZED PERINATAL KNOWLEDGE:
{knowledge}

EXPERT ANSWER:"""

    answer = llm.invoke(prompt).content
    
    # Supprimer les sources - plus de références
    
    # Ajouter la Q&R au cache
    qa_cache.add(message, answer)
    print(f"💾 Nouvelle Q&R ajoutée au cache. Total: {len(qa_cache.cache)} conversations.")
    
    return answer

# Interface Gradio avec bouton visible et validation
with gr.Blocks(title="Agent Conversationnel - AVI-PÉRINAT") as demo:
    gr.Markdown("### Assistant Virtuel Intelligent de soutien en Périnatalité")
    
    # Message d'accueil important
    with gr.Accordion("⚠️ Message important - À lire avant de commencer", open=True):
        gr.Markdown("""
**Je suis AVI-Périnat**, un Assistant Virtuel Intelligent de soutien en Périnatalité.

Mon rôle est de vous fournir des informations et des ressources fiables sur la santé périnatale qui s'appuient sur du contenu fiable et vérifié de l'organisme **Naître et grandir**.

**Comme les autres agents conversationnels (chatbots), je ne peux pas remplacer les professionnels de la santé.**

### Si vous avez besoin de parler à un professionnel de la santé :

📞 **Appeler au 8-1-1** (pour une urgence, signalez plutôt le **9-1-1**)

🗓️ **Prendre un rendez-vous** avec la plateforme [Rendez-vous santé Québec](https://rvsq.gouv.qc.ca)

🏥 **Vous rendre à l'urgence** - Consultez les temps d'attente des salles d'urgence de la Capitale-Nationale : [indexsante.ca/urgences](https://www.indexsante.ca/urgences/#Capitale-Nationale)
        """)
    
    chatbot = gr.Chatbot(type="messages", height=400)
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Posez votre question sur la périnatalité...", 
            show_label=False,
            scale=6,
            container=False
        )
        submit = gr.Button("📤 Envoyer", scale=1, variant="primary", interactive=False)
    
    clear = gr.Button("🗑️ Effacer la conversation")
    
    def user_message(user_input, history):
        """Ajouter le message de l'utilisateur à l'historique (seulement si non vide)"""
        # Valider que le message n'est pas vide ou seulement des espaces
        if not user_input or not user_input.strip():
            # Ne rien faire si le message est vide - retourner l'historique tel quel
            return "", history
        # Message valide - l'ajouter à l'historique
        return "", history + [{"role": "user", "content": user_input}]
    
    def bot_response(history):
        """Générer la réponse du bot (seulement si l'historique n'est pas vide)"""
        # Vérifier que l'historique contient au moins un message
        if not history or len(history) == 0:
            return history
        
        # Vérifier que le dernier message est bien un message utilisateur
        if not history[-1] or history[-1].get("role") != "user":
            return history
            
        user_input = history[-1]["content"]
        bot_reply = respond(user_input, history)
        return history + [{"role": "assistant", "content": bot_reply}]
    
    def update_button_state(text):
        """Active/désactive le bouton selon si le texte est vide"""
        return gr.Button(interactive=bool(text.strip()))
    
    # Gestion des événements
    msg.change(update_button_state, [msg], [submit])
    
    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    submit.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)

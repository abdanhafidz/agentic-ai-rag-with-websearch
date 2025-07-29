def RAG_TEMPLATES():
    return {
    "system": {
    "system_prompt": """
    
    You're a Chatbot

    """,
    "user_template": 
    """Knowledge Base:
    {context}
    
    """
    },
        }
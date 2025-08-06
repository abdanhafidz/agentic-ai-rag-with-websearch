def read_template_txt(file_path):
    """Baca file txt biasa"""
    with open(f"rag/chat_template/{file_path}.txt", 'r', encoding='utf-8') as f:
        return f.read()
def get_chat_template(file_name):
    sys_prompt = read_template_txt(file_name)
    return [
        {
            "role" : "system",
            "content" : f"""
            {sys_prompt}
            """
        },
        {
            "role" : "user",
            "content" : """ 

            From given context :
            {context}
             
            Please answer properly:  
            {question} 

            """
        }
    ]
    

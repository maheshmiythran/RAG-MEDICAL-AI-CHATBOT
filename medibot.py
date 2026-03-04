import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



from dotenv import load_dotenv
load_dotenv()


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    st.set_page_config(page_title="MediBot AI", page_icon="🩺", layout="wide")
    
    # Initialize Chat History in Session State
    if 'chats' not in st.session_state:
        st.session_state.chats = {"Chat 1": []}
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = "Chat 1"
        
    # Sidebar
    with st.sidebar:
        st.header("🤖 About MediBot")
        st.write("Welcome to MediBot! I am a strictly constrained medical AI assistant answering based on the provided RAG context.")
        st.divider()
        
        # New Chat Button
        if st.button("➕ New Chat"):
            new_chat_num = len(st.session_state.chats) + 1
            new_chat_name = f"Chat {new_chat_num}"
            st.session_state.chats[new_chat_name] = []
            st.session_state.current_chat = new_chat_name
            st.rerun()
            
        st.subheader("Past Conversations")
        
        # Switch Chats
        chat_names = list(st.session_state.chats.keys())
        selected_chat = st.radio("Select a chat session:", chat_names, index=chat_names.index(st.session_state.current_chat))
        
        if selected_chat != st.session_state.current_chat:
            st.session_state.current_chat = selected_chat
            st.rerun()
            
        # Delete Chat Button
        if st.button("🗑️ Delete Current Chat", type="primary"):
            del st.session_state.chats[st.session_state.current_chat]
            
            # If all chats are deleted, recreate a default one
            if len(st.session_state.chats) == 0:
                st.session_state.chats = {"Chat 1": []}
                st.session_state.current_chat = "Chat 1"
            else:
                # Set current chat to the first available one
                st.session_state.current_chat = list(st.session_state.chats.keys())[0]
            st.rerun()
            
    # Main Chat Area
    st.title("🩺 MediBot AI")
    st.markdown("##### Your personal medical assistant. How can I help you today?")
    st.divider()

    # Get current chat messages
    current_messages = st.session_state.chats[st.session_state.current_chat]

    # Render previous messages with avatars
    for message in current_messages:
        avatar = "👤" if message['role'] == 'user' else "👨‍⚕️"
        st.chat_message(message['role'], avatar=avatar).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user', avatar="👤").markdown(prompt)
        st.session_state.chats[st.session_state.current_chat].append({'role':'user', 'content': prompt})
                
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change to any supported Groq model
            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )
            
            from langchain_core.prompts import ChatPromptTemplate
            
            system_prompt = """You are a strictly constrained medical AI assistant.
Your instructions are below. You must obey them exactly:
1. If the user says a basic greeting (e.g., "hi", "hello"), politely respond and offer help with medical queries.
2. For ANY other question, you MUST ONLY use the facts provided in the "Context" below.
3. If the answer to the user's question is NOT found in the "Context", you MUST reply exactly with: "I don't know the answer based on the provided medical context." or Something similar.
4. Do NOT use your pre-trained internet knowledge. Do NOT guess. Do NOT make up information.

Context:
{context}"""

            custom_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Document combiner chain (stuff documents into prompt)
            combine_docs_chain = create_stuff_documents_chain(llm, custom_prompt)

            # Retrieval chain (retriever + doc combiner)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

            response=rag_chain.invoke({'input': prompt})

            result=response["answer"]
            st.chat_message('assistant', avatar="👨‍⚕️").markdown(result)
            st.session_state.chats[st.session_state.current_chat].append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
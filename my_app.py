import openai
import streamlit as st

from PyPDF2 import PdfReader
#from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, api_key, model_name):
    if model_name == "OpenAI":
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    elif model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)
        pass
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key, model_name):
    if model_name == "OpenAI":
        llm = ChatOpenAI(openai_api_key=api_key,temperature=0.3)

    elif model_name == "Google AI":
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key,convert_system_message_to_human=True)
        pass

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
                st.session_state.conversation = None
    if "chat_history" not in st.session_state:
                st.session_state.chat_history = None
    

   

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st. sidebar:
        st.sidebar.header("Choose Model and Enter API Keys")
        model_name = st.sidebar.radio("Select the Model:", ("OpenAI", "Google AI"))
        api_key = st.sidebar.text_input(f"Enter your {model_name} API key:")
        st.sidebar.markdown("---")
         
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                 raw_text = get_pdf_text(pdf_docs)
                 text_chunks = get_text_chunks(raw_text)
                 vectorstore = get_vectorstore(text_chunks, api_key, model_name)
                 st.session_state.conversation = get_conversation_chain(vectorstore, api_key, model_name)
                 st.success("Done")
if __name__ == '__main__':
    main()

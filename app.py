import os
import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
import google.generativeai as genai
#from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# This is the first API key input; no need to repeat it in the main function.

# Processing Logic
# PDF load and Split
#def get_pdf_text(pdf_docs):
#  pdfloader = PyPDFLoader(pdf_docs)
#  pdfpages = pdfloader.load_and_split()
#  print(pdfpages[1].page_content)
#  return pdfpages

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Text Splitter
def get_text_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  texts = text_splitter.split_text(text)
  return texts

def get_vector_store(text_chunks, api_key):
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
  vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  # .as_retriever(search_kwargs={"k":5})
  vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the document", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

#Streamlit UI
st.set_page_config(page_title="RAG Application by APJ",
                   layout="wide",
                   page_icon="🧑‍⚕️")

st.header("AI Experiments by APJ💁")
st.subheader("RAG Application")
    
# Step 1: API Key from User
api_key = st.text_input("Step 1: Enter your Google API Key:", type="password", key="api_key_input")


def main():
    
    
    # Step 2: Upload Reference Document
    pdf_docs = st.file_uploader("Step 2: Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
    if st.button("Load Document", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")
    
    # Step 3: Take user question
    user_question = st.text_input("Step 3: Enter your question and Hit Enter", key="user_question")
 

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

  
    #with st.sidebar:
    #   st.title("Step 1")
    #    pdf_docs = st.file_uploader("Upload your PDF Files and Click on Load Button", accept_multiple_files=True, key="pdf_uploader")
    #    if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
    #        with st.spinner("Processing..."):
    #            raw_text = get_pdf_text(pdf_docs)
    #            text_chunks = get_text_chunks(raw_text)
    #            get_vector_store(text_chunks, api_key)
    #            st.success("Done")
# pdfpages = get_pdf_text(pdf_docs)
  #              texts = get_text_chunks(pdfpages)
   #             get_vector_store(texts, GOOGLE_API_KEY)

if __name__ == "__main__":
    main()

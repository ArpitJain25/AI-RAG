import os
import streamlit as st
from streamlit_option_menu import option_menu

import google.generativeai as genai
#from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set page configuration
st.set_page_config(page_title="RAG Application by APJ",
                   layout="wide",
                   page_icon="🧑‍⚕️")

from IPython.display import display
from IPython.display import Markdown
import textwrap


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# sidebar for navigation
with st.sidebar:
      selected = option_menu('Menu',
                           ['About Me',
                            'BJP Manifesto'],
                           menu_icon='rocket',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# This is the first API key input; no need to repeat it in the main function.
GOOGLE_API_KEY = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# Processing Logic
# PDF load and Split
def get_pdf_text(pdf_docs):
  pdfloader = PyPDFLoader(pdf_docs)
  pdfpages = pdfloader.load_and_split()
  print(pdfpages[1].page_content)
  return pdfpages

# Text Splitter
def get_text_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  context = "\n\n".join(str(p.page_content) for p in pdfpages)
  texts = text_splitter.split_text(context)
  return texts

def get_vector_store(text_chunks, GOOGLE_API_KEY):
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
  vector_store = Chroma.from_texts(texts, embeddings,persist_directory = persist_directory).as_retriever(search_kwargs={"k":5})
  client = chromadb.PersistentClient(path="/path/to/save/to")


def get_conversational_chain():
  model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.5)

  template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
  {context}
  Question: {question}
  Helpful Answer:"""
  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
  qa_chain = RetrievalQA.from_chain_type(
      llm=model,
      retriever=vector_store,
      return_source_documents=True,
      chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
  )
  return qa_chain

  
def user_input(user_question, GOOGLE_API_KEY):
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
  new_db = FAISS.load_local("faiss_index", embeddings)
  docs = new_db.similarity_search(user_question)
  chain = get_conversational_chain()
  result = qa_chain.invoke({"query": user_question})
  Markdown(result["result"])
  st.write("Reply: ", result["output_text"])

def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and GOOGLE_API_KEY:  # Ensure API key and user question are provided
        user_input(user_question, GOOGLE_API_KEY)

  
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and GOOGLE_API_KEY:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, GOOGLE_API_KEY)
                st.success("Done")

if __name__ == "__main__":
    main()

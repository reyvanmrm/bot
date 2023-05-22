import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    Upload your PDF to talk
    ''')
 
load_dotenv()
 
def main():
    st.header("Chat with PDF 💬")
 
    # Ask for the OpenAI API Key
    OPENAI_API_KEY = st.text_input("Please input your OpenAI API Key:")
 
    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    if pdf is not None and OPENAI_API_KEY != '':
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
 
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
 
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(model_name='text-davinci-003', api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()

import streamlit as st
import os
import requests
from io import BytesIO
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def read_pdf_from_url(url):
    response = requests.get(url)
    pdf = BytesIO(response.content)
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def main():
    st.header("Chat with PDF 💬")

    # Ask for the OpenAI API Key
    OPENAI_API_KEY = st.text_input("Please input your OpenAI API Key:")

    # Provide a URL to your PDF file
    url = st.text_input("Provide a URL to your PDF file:")

    if url and OPENAI_API_KEY != '':
        text = read_pdf_from_url(url)
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
 
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(model_name='text-davinci-003', api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)
 
if __name__ == '__main__':
    main()

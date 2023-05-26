import requests
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from io import BytesIO
import tempfile

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    Upload your PDFs to talk
    ''')

load_dotenv()

def main():
    st.header("Chat with PDFs 💬")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    pdfs = st.file_uploader("Upload your PDFs", type='pdf', accept_multiple_files=True)
    text = ""

    if pdfs is not None:
        for pdf in pdfs:
            pdf_reader = PdfReader(pdf)
            
            for page in pdf_reader.pages:
                text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = 'multiple_pdfs'
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        queries = st.text_input("Ask questions about your PDF files (separate questions with a semicolon):")
        queries = [query.strip() for query in queries.split(';')]

        for query in queries:
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

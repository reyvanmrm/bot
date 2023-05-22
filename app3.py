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
    Upload your PDF to talk
    ''')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    pdf = st.file_uploader("Upload your PDF", type='pdf')
    pdf_url = st.text_input("Or, enter PDF URL:")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
    elif pdf_url != "":
        pdf_data = requests.get(pdf_url)
        pdf_file = tempfile.NamedTemporaryFile(delete=False)
        pdf_file.write(pdf_data.content)
        pdf_file.close()
        pdf_reader = PdfReader(pdf_file.name)

    if 'pdf_reader' in locals():  # Make sure PDF has been uploaded or downloaded
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4] if pdf is not None else pdf_url.split('/')[-1].split('.')[0]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

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

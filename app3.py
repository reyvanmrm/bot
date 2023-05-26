import requests
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import tempfile

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    Upload your PDFs or input PDF URLs to chat with them
    ''')

load_dotenv()

def main():
    st.header("Chat with PDFs 💬")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    uploaded_pdfs = st.file_uploader("Upload your PDFs", type='pdf', accept_multiple_files=True)
    pdf_urls_input = st.text_input("Or, enter PDF URLs (separate URLs with a semicolon):")
    store_name = st.text_input("Enter a name for your PDFs:")
    uploaded_pkl = st.file_uploader("Upload a saved .pkl file", type='pkl')

    saved_pdf_urls = []
    if uploaded_pkl is not None:
        loaded_data = pickle.load(uploaded_pkl)
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            VectorStore, saved_pdf_urls = loaded_data
        else:
            VectorStore = loaded_data  # fallback for old .pkl files that don't include the URL list
            saved_pdf_urls = []
    else:
        if pdf_urls_input != "":
            pdf_urls = [url.strip() for url in pdf_urls_input.split(";")]
            saved_pdf_urls.extend(pdf_urls)
            pdfs = []
            for pdf_url in pdf_urls:
                pdf_data = requests.get(pdf_url)
                pdf_file = tempfile.NamedTemporaryFile(delete=False)
                pdf_file.write(pdf_data.content)
                pdf_file.close()
                pdfs.append(PdfReader(pdf_file.name))
        else:
            pdfs = [PdfReader(uploaded_pdf) for uploaded_pdf in uploaded_pdfs]

        text = ""
        for pdf in pdfs:
            for page in pdf.pages:
                text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump((VectorStore, saved_pdf_urls), f)

    st.write(f'Loaded PDFs: {store_name}')

    queries = st.text_input("Ask questions about your PDF files (separate questions with a semicolon):")

    for query in queries.split(';'):
        docs = VectorStore.similarity_search(query=query.strip(), k=3)

        llm = OpenAI(model_name='text-davinci-003', api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query.strip(), max_tokens=500)
            print(cb)
        st.write(response)

    with open(f"{store_name}.pkl", "rb") as f:
        bytes = f.read()
        st.download_button(label=f"Download {store_name}.pkl", data=bytes, file_name=f"{store_name}.pkl", mime='application/octet-stream')

if __name__ == '__main__':
    main()


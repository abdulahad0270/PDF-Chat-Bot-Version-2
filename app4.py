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

# Function to initialize session state
def init_session_state():
    return st.session_state.setdefault('history', [])

# Sidebar contents

with st.sidebar:
    st.title("PDF Chat Bot")
    if st.button("New Chat"):
        st.session_state.history = []
        st.experimental_rerun()


with st.sidebar:
    #st.title("PDF Chat Bot")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made by Abdul Ahad')



def main():
    st.header("Chat with PDF")

    load_dotenv()

    # Initialize session state
    if 'history' not in st.session_state:
        init_session_state()

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.write(pdf.name)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask Questions about your PDF file:")
        st.write(query)

        #if st.button("New Chat"):
            # Clear history and reset conversation
            #st.session_state.history = []
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.session_state.history.append({'query': query, 'response': response})
                print(cb)
            st.write(response)

        # Display conversation history
        with st.sidebar:
            if st.session_state.history:
                st.subheader("Conversation History")
                for entry in st.session_state.history:
                    st.write(f"User: {entry['query']}")
                    st.write(f"Bot: {entry['response']}")
                    st.write("---")


if __name__ == '__main__':
    main()

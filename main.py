import streamlit as st
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt):
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        api_key=groq_api_key
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    return chain.invoke(docs)

st.set_page_config(
    page_title = "Writing Text Summarization"
)
st.title("Writing Text Summarization")

txt_input = st.text_area(
    "Enter your text",
    "",
    height=200
)

result = []
with st.form("summarize_form", clear_on_submit=True):
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        disabled=not txt_input
    )
    submitted = st.form_submit_button("Submit")
    if submitted and groq_api_key.startswith("gsk_"):
        response = generate_response(txt_input)
        result.append(response)
        del groq_api_key

if len(result):
    st.info(response["output_text"])
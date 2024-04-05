import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from io import BytesIO
import openai
from PyPDF2 import PdfReader

st.set_page_config(page_title="Chat with the Bain Report (M&A)", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Bain Reports (M&A, PE & Tech)")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Bain Report!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data(file_contents):
    with st.spinner(text="Processing the uploaded PDF file..."):
        text = ""
        pdf_reader = PdfReader(BytesIO(file_contents))
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Initialize the ServiceContext with OpenAI model
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4", temperature=0.8, system_prompt="You are an expert on the Bain Reports. Please provide detailed insights from the Bain Reports on [specific topic or question]."))

        # Create a single document from the PDF content
        doc = Document(content=text, source="uploaded_pdf")

        # Create a VectorStoreIndex containing the single document
        index = VectorStoreIndex(nodes=[doc], service_context=service_context)

        return index

# New method for uploading PDF file
uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
if uploaded_file is not None:
    file_contents = uploaded_file.read()
    st.session_state.file_name = uploaded_file.name
    index = load_data(file_contents)
    print('loadedindex',index)
else:
    st.session_state.file_name = None

# Check if PDF file is uploaded
if st.session_state.file_name:
    st.write(f"PDF file '{st.session_state.file_name}' is uploaded.")

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history
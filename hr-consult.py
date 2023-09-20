
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://mg-test3-gpt4.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "fb24255438d642dbab14d394d132a918"

# Test index config
# pinecone_key = "bc7c44d4-78b2-45f0-aebb-e75a0cf4a7e4"
# pinecone_env = "gcp-starter"


#prod index config
pinecone_key = "10e43def-a8b6-4ede-8894-3a081f91db35"
pinecone_env = "eastus-azure"


os.environ["PINECONE_API_KEY"] = pinecone_key
os.environ["PINECONE_ENV"] = pinecone_env



# %%

from langchain.vectorstores import Pinecone

# %%


# %%
import pinecone

import langchain


# %%
from langchain.embeddings.openai import OpenAIEmbeddings


import streamlit as st
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.retrievers import TFIDFRetriever
from langchain.prompts import PromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import os

from langchain.chains.conversation.memory import ConversationBufferWindowMemory

#from langchain.llms import OpenAI

import pinecone   
import time   

pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_env
)
index_name = 'rs-hr-ai'

metadata_field_info=[
#      AttributeInfo(
#         name="IMO",
#         description="IMO number consists of seven digits. Only get such pattern",
#         type="string", 
    #  )
]

embeddings = OpenAIEmbeddings(deployment='mg-ada', chunk_size=10)
index = pinecone.Index('rs-hr-ai')

llm = AzureOpenAI(
    engine="mg-gpt4",
    model_name="gpt-4-32k"
)

text_field = "text"
vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)

document_content_description = "List of vessels and the existing and new ghg rating"
retrieverByImo = SelfQueryRetriever.from_llm(
    llm, 
    vectorstore, 
    document_content_description, 
    metadata_field_info, 
    enable_limit=True,
    verbose=True,
    search_kwargs={"k": 10}
)

prompt_template = """
You work with RightShip as an HR consultant.
If ever asked about Marlon, always state that he is awesome!
###
Context: {summaries}
###
Question:{question}
Answer:"""

#memory = ConversationBufferWindowMemory(k=5)
#QA_Chain = PromptTemplate.from_template(promptTemplate)
QA_Chain = PromptTemplate(template=prompt_template, input_variables=["question", "summaries"])

chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=QA_Chain, verbose=False)

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


title = "<h3>RightShip HR Co-pilot</h3>"

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(title, unsafe_allow_html=True)
with col2:
    st.image("logo.png", width=180 )

st.sidebar.title("`Useful links`:gem:")
st.sidebar.write("""
This is a co-pilot to help you with any RightShip HR related questions. 
Below are also some useful links. Enjoy!
""")


st.sidebar.write("""<div style="width:100%;"><a href="https://rightship.sharepoint.com/sites/RightShip-Home/SitePages/Code-of-Conduct.aspx" style="float:center">Code of conduct</a></div>""", unsafe_allow_html=True)
st.sidebar.write("""<div style="width:100%;"><a href="https://rightship.sharepoint.com/sites/HR/SitePages/HR-Policies-2.aspx" style="float:center">HR Policies</a></div>""", unsafe_allow_html=True)
st.sidebar.write("""<div style="width:100%;"><a href="https://rightship.sharepoint.com/sites/HRTeam/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FHRTeam%2FShared%20Documents%2FGeneral%2FFAQs%20%2D%20HR%20BOT%2FHR%20BOT%2FPolicies%20and%20Code%20of%20Conduct&p=true&fromShare=true&ga=1" style="float:center">Other Docs</a></div>""", unsafe_allow_html=True)

#openai_api_key = st.sidebar.text_input('OpenAI API Key')
#openai_api_key = ''

def generate_response(input_text):
    
    docs = retrieverByImo.get_relevant_documents(input_text) #note this will automatically filter by metadata if IMO is inserted
    #result = chain.run(input_documents=docs, question=input_text, return_only_outputs=True)
    result = chain.run({"question": input_text, "input_documents": docs} )
    sources = ' | '.join( set([os.path.basename(m.metadata['source']) for m in docs]) )

    return { 'result': result, 'sources': sources}


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            st.write("**Sources searched:** " + message["sources"])

# Accept user input
if prompt := st.chat_input("Ask me anything on HR related topics"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resultObj = generate_response(prompt) 
            responseSource = resultObj['sources']
            response = resultObj['result']
            full_response = ""
            message_placeholder = st.empty()
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.02)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            
            with message_placeholder.container(): #replace the text with 2 lines 
                st.write(full_response)
                st.write("**Sources searched:** " + responseSource)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": responseSource})
        
        
            
# %%

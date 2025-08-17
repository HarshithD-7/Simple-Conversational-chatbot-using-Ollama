import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"



prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Please respond to the question asked.\n\nQuestion: {question}"
)




# Prompt Template
# prompt = ChatPromptTemplate.from_message(
#     [
#         ("sysytem", "You are a helpful assitant. Please respond to the questions asked"),
#         ("user", "Question:{question}")
#     ]
# )


# Streamlit UI
st.title("Langchain Demo With Gemma Model")
input_text = st.text_input("What question do you have in mind?")

# Ollama Gemma model
llm = Ollama(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Handle user input
if input_text:
    st.write(chain.invoke({"question": input_text}))
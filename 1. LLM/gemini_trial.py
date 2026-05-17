import streamlit as st


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def response():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY") 
    )

    response = llm.invoke("Who is elon musk in one sentence?")
    return response

if __name__ == "__main__":
    st.title("Gemini-3-Flash-Preview Trial")

    st.write("Response from Gemini-3-Flash-Preview:")

    response_text = response()
    st.write(response_text.content)

    st.write("Response metadata:")
    st.write(response_text.usage_metadata)

    st.write("Overall Response")
    st.write(response_text)
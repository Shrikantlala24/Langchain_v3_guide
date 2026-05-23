# report summarizer agent using sequential chain
# topic -> LLM -> Report
#                    |-> LLM -> Summary

# Import LLM 1
from langchain_google_genai import ChatGoogleGenerativeAI

# Import LLM 2
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Import prompt + Output parser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# extras
from dotenv import load_dotenv
import pprint

load_dotenv()

prompt1 = PromptTemplate(
    input_variables=['topic'],
    template=' Generate a detailed report on {topic}'
)

llm1 = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash'
)

prompt2 = PromptTemplate(
    input_variables=['report'],
    template=' Summarize the following report in at most three lines: {report}'
)

llm_info = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text_generation'
)

llm2 = ChatHuggingFace(llm=llm_info)


parser = StrOutputParser()

seq_chain = prompt1 | llm1 | parser | prompt2 | llm2 | parser

response = seq_chain.invoke('attention is all you need- research paper')

print(response)

seq_chain.get_graph().print_ascii()
    
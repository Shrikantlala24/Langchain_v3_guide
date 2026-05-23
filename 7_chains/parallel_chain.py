# report summarizer agent using sequential chain
#                    topic
#                      |
#                     LLM1
#                      |
#                    Report
#                    /   \   
#                LLM1     LLM2
#                 |         |
#               Notes      Quiz
#                  \       /
#                    merge
#                      |
#                    LLM3
#                      |
#                   response


# Import LLM 1
from langchain_google_genai import ChatGoogleGenerativeAI

# Import LLM 2
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Import prompt + Output parser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# runnable parallel chain method
from langchain_core.runnables import RunnableParallel

# extras
from dotenv import load_dotenv
import pprint

load_dotenv()

prompt1 = PromptTemplate(
    input_variables=['topic'],
    template="Generate a detailed report on {topic}"
)

prompt2 = PromptTemplate(
    input_variables=['report'],
    template="Create Notes using this report {report}"
)

prompt3 = PromptTemplate(
    input_variables=['report'],
    template="Create a quiz using this report {report}"
)

prompt4 = PromptTemplate(
    input_variables=['notes', 'quiz'],
    template="Merge the following notes and quiz into a single response. Notes: {notes} Quiz: {quiz}"
)

llm1 = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash'
)

llm_info = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text_generation'
)

llm2 = ChatHuggingFace(llm=llm_info)

parser = StrOutputParser()

chain1 = prompt1 | llm1 | parser


parallel_chain = RunnableParallel({    
    'notes': prompt2 | llm1 | parser,
    'quiz': prompt3 | llm2 | parser
})

chain4 = prompt4 | llm2 | parser

complete_chain = chain1 | parallel_chain | chain4

response = complete_chain.invoke('attention is all you need- research paper')

print(response)

complete_chain.get_graph().print_ascii()

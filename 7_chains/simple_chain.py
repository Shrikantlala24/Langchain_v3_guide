from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    input_variables=['question'],
    template=' answer the following question with whole meaning of it in at most three lines: {question}'
)

model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash'
)

parser = StrOutputParser()


chain = prompt | model | parser

response = chain.invoke('what is deep learning?')

print(response)

chain.get_graph().print_ascii()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Deep learning is a subset of machine learning inspired by the human brain, utilizing multi-layered artificial neural networks to automatically learn from vast data. 
# By extracting complex patterns and representations without human intervention, it enables computers to solve highly sophisticated, unstructured problems. 
# It serves as the core technology behind modern AI breakthroughs, including natural language processing, computer vision, and autonomous systems.
#       +-------------+      
#       | PromptInput |      
#       +-------------+      
#              *             
#              *             
#              *             
#     +----------------+     
#     | PromptTemplate |     
#     +----------------+     
#              *             
#              *             
#              *             
# +------------------------+ 
# | ChatGoogleGenerativeAI | 
# +------------------------+ 
#              *             
#              *             
#              *             
#     +-----------------+    
#     | StrOutputParser |    
#     +-----------------+    
#              *             
#              *             
#              *             
# +-----------------------+  
# | StrOutputParserOutput |  
# +-----------------------+  
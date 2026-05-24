# Import LLM 1
from langchain_google_genai import ChatGoogleGenerativeAI
# Import LLM 2
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# Import prompt + Output parser
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Runnable ke imports
from langchain_core.runnables import RunnableLambda, RunnableBranch

# Pydantic ke imports
from pydantic import BaseModel, Field
from typing import Literal, Annotated

# extras
from dotenv import load_dotenv
import pprint
load_dotenv()

llm1 = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash'
)

llm_info = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text_generation'
)
llm2 = ChatHuggingFace(llm=llm_info)


class Feedback(BaseModel):

    sentiment: Annotated[Literal['positive', 'negative'], Field(description='Give the sentiment of the feedback')]


parser1 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser1.get_format_instructions()}
)

# Here is the classifer, which will give away the sentiment 
classifier_chain = prompt1 | llm1 | parser1

# Now, we'll create two prompts, one for positive feedback and one for negative feedback. These will be used in the conditional chain.
# these prompts will be used to generate responses based on the sentiment of the feedback.

positive_prompt = PromptTemplate(
    template='The following feedback is positive: {feedback}. Generate a response that thanks the user for their positive feedback and encourages them to continue using the product.',
    input_variables=['feedback']
)

negative_prompt = PromptTemplate(
    template='The following feedback is negative: {feedback}. Generate a response that apologizes for the negative experience and offers assistance to resolve the issue.',
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (RunnableLambda(lambda x:x.sentiment == 'positive'),positive_prompt | llm2),
    (RunnableLambda(lambda x:x.sentiment == 'negative'),negative_prompt | llm2),
    RunnableLambda(lambda x: 'Invalid sentiment')
)

final_chain  = classifier_chain | branch_chain

response = final_chain.invoke('The iPhone 15 Pro Max is an absolute disaster for my savings but a complete masterpiece in my hands; I desperately wanted to hate this overpriced brick, yet its flawlessly smooth 120Hz display and absurdly long battery life make it impossible to regret, which honestly just infuriates me.')

print(response)

final_chain.get_graph().print_ascii()
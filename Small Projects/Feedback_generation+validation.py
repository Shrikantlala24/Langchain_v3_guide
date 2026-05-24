# Import LLM 1
from langchain_google_genai import ChatGoogleGenerativeAI
# Import LLM 2
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# Import prompt + Output parser
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
# Runnable functions
from langchain_core.runnables import RunnableLambda


# extras
from dotenv import load_dotenv
import pprint

# Also pydantic for structured output parsing
from pydantic import BaseModel, Field
from typing import Literal, Annotated

load_dotenv()

# we'll be building a conditional sentiment analysis chain.
# LLM1 (gemini) will generate a feedback object with two fields: feedback and sentiment.
# LLM2 (llama) will take the feedback, and will check the sentiment on it's own without seeing the sentiment field give by LLM1.

# this way we'll check what's the difference between Gemini and HuggingFace's Llama 3.1 in sentiment analysis.

# if the sentiment is positive -> we'll give user a feedback form (only a dummy URL as in print)
# if the sentiment is negative -> we'll give user a contact support form ( Same here, only a dummy URL as in print)

llm1 = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash'
)
llm_info = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text_generation'
)
llm2 = ChatHuggingFace(llm=llm_info)

class Feedback(BaseModel):
    feedback: Annotated[str, Field(description='The feedback given by the user')]
    sentiment: Annotated[Literal['positive', 'negative'], Field(description='The sentiment of the feedback, can be either positive or negative')]

class sentiment_Response(BaseModel):
    sentiment: Annotated[Literal['positive', 'negative'], Field(description='The sentiment of the feedback, can be either positive or negative')]

prompt_feedback_generation = PromptTemplate(
    input_variables=['product'],
    template='You are a helpful assistant. Please provide a dummy but most confusing feedback, which is difficult to identify sentiment. about this product: {product} and also provide the sentiment of the feedback in a JSON format with two fields: feedback and sentiment. The sentiment can be either positive or negative.'
)

structured_output_parser = PydanticOutputParser(pydantic_object=Feedback)  # ✅

feedback_chain = prompt_feedback_generation | llm1.with_structured_output(Feedback)


# Fix 2 & 3: define sentiment_parser first, fix pydantic_object arg
sentiment_parser = PydanticOutputParser(pydantic_object=sentiment_Response)

prompt_classifier = PromptTemplate(
    input_variables=["feedback"],
    template=(
        "You are a helpful assistant. Classify the sentiment of the following feedback as either positive or negative: {feedback}.\n"
        "{format_instructions}"
    ),
    partial_variables={"format_instructions": sentiment_parser.get_format_instructions()},
)

classifier_chain = prompt_classifier | llm2 | sentiment_parser

extract_feedback = RunnableLambda(lambda x: {"feedback": x.feedback})

full_chain = feedback_chain | extract_feedback | classifier_chain

response = full_chain.invoke('iPhone 15 Pro Max')
print("Feedback and sentiment from LLM1 (Gemini):")
pprint.pprint(feedback_chain.invoke('iPhone 15 Pro Max'))
print("\nSentiment from LLM2 (Llama):")
pprint.pprint(response)
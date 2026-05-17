from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()

llm_info = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 200}
)

model = ChatHuggingFace(llm=llm_info)

response = model.invoke("What is Deep learning?")
print(response.content)
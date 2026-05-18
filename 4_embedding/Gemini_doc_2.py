# Not woriking

# import google.genai as genai
# from dotenv import load_dotenv
# import os

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Use the native embedding method
# result = genai.embed_content(
#     model="models/text-embedding-004",  # Correct model name for direct API
#     content=[
#         'what is Deep Learning?',
#         'what is Machine Learning?',
#         'what is Generative AI?'
#     ],
#     task_type="retrieval_document",  # Options: retrieval_query, retrieval_document, semantic_similarity
#     output_dimensionality=3
# )

# embeddings = result['embedding']
# print(len(embeddings))  # Should print 3
# print(len(embeddings[0]))  # Should print 3
# print(embeddings)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pprint import pprint

from typing import TypedDict, Annotated, List, Dict

load_dotenv()  # Load environment variables from .env file

# I am making an Agent, which takes input -> a document (basically a paragraph) and returns the information from that doc.
# that information will be used for Linedin ID creation of that deceloper


class Linkedin_data(TypedDict):
    Name : Annotated[str, 'enter the name of the developer']
    mail_id : Annotated[str, 'Enter the mail ID of developer']
    Website : Annotated[str,'Enter his portfolio website']
    tech_stack : Annotated[List[str],'Enter the techstack of this developer']
    projects : Annotated[Dict[str,str], 'enter the project name along with its small description']
    summary : Annotated[str,'enter the summary']

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash',
)

structured_model = model.with_structured_output(Linkedin_data)

response = structured_model.invoke("""
My name is Aarav Mehta and I am a Backend Developer focused on cloud-native applications and API development. 
I primarily work with Java, Spring Boot, PostgreSQL, Docker, and Kubernetes.

One of my major projects is CloudCart, an e-commerce backend system that handles authentication, payment integration, and order management. 
I also developed DevTrack, a project monitoring dashboard that visualizes deployment metrics and CI/CD pipelines in real time.

I enjoy designing scalable architectures and optimizing backend performance for high-traffic systems.

Email: aarav.mehta.dev@gmail.com
Website: https://aaravmehta.tech
""")

pprint(response)

# This is the reponse I got

# {'Name': 'Aarav Mehta',
#  'Website': 'https://aaravmehta.tech',
#  'mail_id': 'aarav.mehta.dev@gmail.com',
#  'projects': {'CloudCart': 'An e-commerce backend system that handles '
#                            'authentication, payment integration, and order '
#                            'management.',
#               'DevTrack': 'A project monitoring dashboard that visualizes '
#                           'deployment metrics and CI/CD pipelines in real '
#                           'time.'},
#  'summary': 'Backend Developer focused on cloud-native applications and API '
#             'development. I enjoy designing scalable architectures and '
#             'optimizing backend performance for high-traffic systems.',
#  'tech_stack': ['Java', 'Spring Boot', 'PostgreSQL', 'Docker', 'Kubernetes']}
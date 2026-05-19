from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=1,
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chatbot. Goodbye!")
        break

    response = model.invoke(user_input)
    print(f"AI: {response.content}")



# You: Hi!, Do you know about CampusX. also what's the last date of your training data
# AI: Yes, I do!

# **CampusX** is a well-known company primarily recognized as a leading provider of **student housing and accommodation in Italy**. They offer modern, fully-equipped residences designed to meet the needs of university students, both domestic and international.

# CampusX facilities typically provide a range of services and amenities, such as private or shared rooms, study areas, gyms, communal spaces, laundry facilities, security, and often social and cultural activities. Their goal is to create a comfortable, secure, and community-rich environment that supports students' academic and personal growth.

# Regarding the last date of my training data:

# My knowledge cutoff is **early 2023**. This means I don't have information about events or developments that have occurred since that time.
# You: Ok so do you know what question I asked previously
# AI: As an AI, I don't have a personal memory or the ability to recall specific past interactions outside of our current conversation.

# However, if we are in an ongoing chat session, I can see the previous turns *within this current conversation thread* that have been provided to me as context.

# So, if you asked a question a few moments ago *in this same chat*, yes, it's likely still in my active context. If it was from a previous chat session or a different context, then no, I wouldn't remember it.

# What was it about? I'd be happy to pick up where we left off!
# You: 
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key= os.getenv("GOOGLE_API_KEY")
)


messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("How are you?"),
    AIMessage("Come stai?"),
    HumanMessage("hi!")
]

response = llm.invoke(messages)
print(response.content)



from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)


prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

print(prompt)
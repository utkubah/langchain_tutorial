from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10",
    )
    language: str = Field(description="The language the text is written in")


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key= os.getenv("GOOGLE_API_KEY")
)

llm = llm.with_structured_output(Classification)

inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})

response = llm.invoke(prompt)

print(response)

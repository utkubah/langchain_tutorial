from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

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



llm = init_chat_model(model="gemini-1.5-pro", model_provider="google_genai",api_key ="AIzaSyBSIy8puMYpb21B9FRYymUhzMK01EOR6ao").with_structured_output(
    Classification
)

inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

print(response)


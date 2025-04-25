import asyncio
import sqlite3

import requests
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from langgraph.prebuilt import create_react_agent
from langchain import hub
from langchain.chat_models import init_chat_model   
from dotenv import load_dotenv
import os

load_dotenv()



def get_engine_for_chinook_db() -> Engine:
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


async def main() -> None:
    # Create the engine and database wrapper.
    engine = get_engine_for_chinook_db()
    db = SQLDatabase(engine)

    # Create the toolkit
    llm = init_chat_model(model="gemini-2.0-flash-lite", model_provider="google_genai",api_key =os.getenv("GOOGLE_API_KEY"))
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()
    

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

    assert len(prompt_template.messages) == 1
    prompt_template.messages[0].pretty_print()

    system_message = prompt_template.format(dialect="SQLite", top_k=5)

    agent_executor = create_react_agent(llm, tools, prompt=system_message)

    question = "Can you write me a sql query to see country's customers spent the most?"

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()




if __name__ == "__main__":
    asyncio.run(main())

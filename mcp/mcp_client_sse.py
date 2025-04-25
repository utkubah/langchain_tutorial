import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="gemini-1.5-pro", model_provider="google_genai",api_key ="AIzaSyBSIy8puMYpb21B9FRYymUhzMK01EOR6ao")

async def main():
    async with MultiServerMCPClient(
        {
            "mcp_server_sse": {
                "transport": "sse",
                "url": "http://localhost:8000/sse"
            }
        }
    ) as client:

        tools = client.get_tools()
        
        agent = create_react_agent(llm, tools)
        agent_response = await agent.ainvoke({"messages": "what's (3 + 5)*8?"})


        print(agent_response)

if __name__ == "__main__":
    asyncio.run(main())
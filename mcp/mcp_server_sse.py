from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test Server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="sse")




#to test if server is working or not: npx @modelcontextprotocol/inspector python mcp_server.py

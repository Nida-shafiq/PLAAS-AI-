from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def web_legal_search(query: str) -> str:
    """
    Search the web for general or current legal information not covered by the
    local Pakistani legal database (e.g. recent news, non-Pakistani law, general
    legal concepts). Use only when the FAISS legal search does not return a
    relevant result.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Web search failed: {str(e)}"

    if not results:
        return "No web results found."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        formatted.append(f"{i}. {title}\n{body}\nSource: {href}")

    return "\n\n".join(formatted)

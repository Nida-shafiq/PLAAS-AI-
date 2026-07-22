from langchain_core.tools import tool


@tool
def legal_disclaimer(context: str = "") -> str:
    """
    Returns a standard legal disclaimer. Call this at the end of any response
    that provides legal information, interpretation, or guidance, so the user
    understands this is not a substitute for professional legal advice.
    """
    return (
        "Disclaimer: This information is provided for general educational purposes "
        "only and is based on Pakistani legal texts (Constitution, CrPC, Police Order 2002) "
        "and/or web search results. It does not constitute legal advice. For advice on "
        "your specific situation, please consult a licensed lawyer in Pakistan."
    )

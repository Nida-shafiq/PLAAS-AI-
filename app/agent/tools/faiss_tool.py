from langchain_core.tools import tool
from app.rag.retriever import retrieve_documents


@tool
def faiss_legal_search(query: str) -> str:
    """
    Search Pakistani legal documents (Constitution, CrPC, Police Order 2002)
    for relevant clauses, sections, and articles. Use this for any question
    about Pakistani law, criminal procedure, police powers, or constitutional rights.
    """
    results = retrieve_documents(query)
    if not results:
        return "No relevant legal provisions found in the local database."

    formatted = []
    for i, r in enumerate(results, start=1):
        citation = f"[{r.get('source_pdf','?')} p.{r.get('page','?')} clause:{r.get('clause_label','N/A')}]"
        formatted.append(f"{i}. {r['text'].strip()} {citation}")

    return "\n\n".join(formatted)

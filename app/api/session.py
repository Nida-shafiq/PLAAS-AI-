from typing import Dict, List
from langchain_core.messages import BaseMessage

_sessions: Dict[str, List[BaseMessage]] = {}

def get_history(session_id: str) -> List[BaseMessage]:
    return _sessions.setdefault(session_id, [])

def save_history(session_id: str, messages: List[BaseMessage]) -> None:
    _sessions[session_id] = messages

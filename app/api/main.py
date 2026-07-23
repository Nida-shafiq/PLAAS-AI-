from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.agent.graph import agent_graph
from app.api.session import get_history, save_history
from app.core.logger import logger

app = FastAPI(title='PLAAS AI Agentic Legal Research API')

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/chat', response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info(f'session={req.session_id} query={req.message}')
    history = get_history(req.session_id)
    history = history + [HumanMessage(content=req.message)]
    result = agent_graph.invoke({'messages': history})
    updated_messages = result['messages']
    save_history(req.session_id, updated_messages)
    raw_content = updated_messages[-1].content
    if isinstance(raw_content, list):
        answer = ' '.join(block.get('text', '') for block in raw_content if isinstance(block, dict))
    else:
        answer = raw_content
    logger.info(f'session={req.session_id} response_length={len(answer)}')
    return ChatResponse(session_id=req.session_id, response=answer)

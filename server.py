from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agent import app as agent_app
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI(title="AI News Newsletter Agent API")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartRequest(BaseModel):
    goal: str
    hitl_enabled: bool = True

class FeedbackRequest(BaseModel):
    approved: bool
    feedback: Optional[str] = ""

@app.post("/api/start")
async def start_agent(request: StartRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "goal": request.goal,
        "hitl_enabled": request.hitl_enabled,
        "critique_attempts": 0,
        "logs": ["Agent started."]
    }
    
    # We use invoke/stream. Using stream so we don't block entirely, or invoke if we want it to run until breakpoint.
    # We will run it in the background or just synchronously until it hits a breakpoint or END.
    # Synchronous is fine for this demo as the agent steps take a few seconds.
    try:
        # Run until the next breakpoint (human_approval_node) or END
        final_state = agent_app.invoke(initial_state, config)
        return {"thread_id": thread_id, "status": "running or paused", "current_state": final_state}
    except Exception as e:
        return {"thread_id": thread_id, "status": "error", "error": str(e)}

@app.get("/api/status/{thread_id}")
async def get_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = agent_app.get_state(config)
    
    if not state_snapshot:
        raise HTTPException(status_code=404, detail="Thread not found")
        
    return {
        "thread_id": thread_id,
        "values": state_snapshot.values,
        "next": state_snapshot.next, # If this is ('human_approval_node',), it means it's waiting!
        "is_paused": len(state_snapshot.next) > 0
    }

@app.post("/api/feedback/{thread_id}")
async def provide_feedback(thread_id: str, request: FeedbackRequest):
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = agent_app.get_state(config)
    
    if not state_snapshot or not state_snapshot.next:
        raise HTTPException(status_code=400, detail="Agent is not waiting for feedback.")
        
    # Update the state with the human feedback
    agent_app.update_state(
        config,
        {
            "human_approved": request.approved,
            "human_feedback": request.feedback,
            "logs": state_snapshot.values.get("logs", []) + [f"Human feedback received: Approved={request.approved}"]
        }
    )
    
    # Resume the graph by invoking with None (it continues from where it paused)
    try:
        final_state = agent_app.invoke(None, config)
        return {"thread_id": thread_id, "status": "resumed", "current_state": final_state}
    except Exception as e:
        return {"thread_id": thread_id, "status": "error", "error": str(e)}

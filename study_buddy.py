import os
import time
import json
import uuid
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("study_buddy")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

METRICS = {
    "plans_generated": 0,
    "quizzes_generated": 0,
    "evaluations_done": 0,
    "sessions_created": 0,
}

class FileTool:
    def __init__(self, base_path="study_buddy_store"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save_json(self, name: str, data: dict) -> str:
        path = os.path.join(self.base_path, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def load_json(self, name: str) -> Optional[dict]:
        path = os.path.join(self.base_path, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

file_tool = FileTool()

class MemoryBank:
    def __init__(self, filename="memory_bank.json"):
        self.filename = os.path.join(file_tool.base_path, filename)
        if not os.path.exists(self.filename):
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump({"records": []}, f)

    def add_record(self, record: dict):
        with open(self.filename, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["records"].append(record)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()

    def query_recent(self, limit=5) -> List[dict]:
        with open(self.filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["records"][-limit:]

memory = MemoryBank()

class SessionService:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "messages": [],
            "created_at": time.time()
        }
        METRICS["sessions_created"] += 1
        return session_id

    def append_message(self, session_id: str, role: str, content: str):
        self.sessions[session_id]["messages"].append({
            "role": role,
            "content": content,
            "ts": time.time()
        })

    def get_messages(self, session_id: str) -> List[dict]:
        return self.sessions[session_id]["messages"]

sessions = SessionService()

def compact_context(messages: List[dict]) -> List[dict]:
    if not messages:
        return []
    chosen = messages[-10:]
    merged = " ".join(m["content"] for m in chosen)
    compacted = [{"role": "system", "content": f"COMPACTED_HISTORY: {merged[:2000]}"}]
    compacted += chosen[-3:]
    return compacted

try:
    import openai
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    OPENAI_AVAILABLE = False

def llm_chat(messages: List[dict], temperature=0.2, max_tokens=512) -> str:
    if OPENAI_AVAILABLE:
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    joined = " ".join(m["content"] for m in messages[-3:])
    return f"[MOCK RESPONSE BASED ON: {joined[:200]}]"

class AgentMessage:
    def __init__(self, sender: str, recipient: str, payload: dict):
        self.sender = sender
        self.recipient = recipient
        self.payload = payload
        self.id = str(uuid.uuid4())
        self.ts = time.time()

class Agent:
    def __init__(self, name: str):
        self.name = name

    def handle(self, msg: AgentMessage) -> Tuple[bool, Optional[AgentMessage]]:
        raise NotImplementedError

class PlannerAgent(Agent):
    def handle(self, msg: AgentMessage) -> Tuple[bool, Optional[AgentMessage]]:
        topic = msg.payload.get("topic")
        hours = msg.payload.get("hours", 5)
        pace = msg.payload.get("pace", "moderate")
        prompt = [
            {"role": "system", "content": "You are a study planner."},
            {"role": "user", "content": f"Create a {hours}-hour study plan for {topic}. Pace={pace}."}
        ]
        reply = llm_chat(prompt)
        METRICS["plans_generated"] += 1
        plan_id = f"plan_{int(time.time())}"
        file_tool.save_json(plan_id, {"topic": topic, "plan": reply})
        memory.add_record({"type": "plan", "topic": topic, "ts": time.time()})
        return True, AgentMessage(self.name, msg.sender, {"plan": reply})

class ContentGenAgent(Agent):
    def handle(self, msg: AgentMessage) -> Tuple[bool, Optional[AgentMessage]]:
        topic = msg.payload.get("topic")
        prompt = [
            {"role": "system", "content": "You generate concise study notes."},
            {"role": "user", "content": f"Create study notes for {topic}."}
        ]
        reply = llm_chat(prompt)
        memory.add_record({"type": "note", "topic": topic, "ts": time.time()})
        return True, AgentMessage(self.name, msg.sender, {"note": reply})

class QuizMasterAgent(Agent):
    def handle(self, msg: AgentMessage) -> Tuple[bool, Optional[AgentMessage]]:
        topic = msg.payload.get("topic")
        note = msg.payload.get("note")
        prompt = [
            {"role": "system", "content": "You generate quizzes."},
            {"role": "user", "content": f"Create a quiz from this note:\n{note}"}
        ]
        reply = llm_chat(prompt)
        METRICS["quizzes_generated"] += 1
        memory.add_record({"type": "quiz", "topic": topic, "ts": time.time()})
        return True, AgentMessage(self.name, msg.sender, {"quiz": reply})

class EvaluatorAgent(Agent):
    def handle(self, msg: AgentMessage) -> Tuple[bool, Optional[AgentMessage]]:
        question = msg.payload.get("question")
        answer = msg.payload.get("answer")
        prompt = [
            {"role": "system", "content": "You evaluate student answers."},
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}
        ]
        reply = llm_chat(prompt, temperature=0)
        METRICS["evaluations_done"] += 1
        memory.add_record({"type": "evaluation", "ts": time.time()})
        return True, AgentMessage(self.name, msg.sender, {"evaluation": reply})

class OrchestratorAgent(Agent):
    def __init__(self, name: str, agents: Dict[str, Agent]):
        super().__init__(name)
        self.agents = agents

    def dispatch(self, agent_name: str, msg: AgentMessage) -> AgentMessage:
        _, response = self.agents[agent_name].handle(msg)
        return response

    def handle_user_request(self, session_id: str, payload: dict) -> dict:
        topic = payload["topic"]
        planner_resp = self.dispatch("Planner", AgentMessage(self.name, "Planner", payload))
        content_resp = self.dispatch("ContentGen", AgentMessage(self.name, "ContentGen", {"topic": topic}))
        quiz_resp = self.dispatch(
            "QuizMaster",
            AgentMessage(self.name, "QuizMaster", {"topic": topic, "note": content_resp.payload["note"]})
        )
        artifact = {
            "topic": topic,
            "plan": planner_resp.payload["plan"],
            "note": content_resp.payload["note"],
            "quiz": quiz_resp.payload["quiz"],
            "ts": time.time()
        }
        artifact_id = f"artifact_{int(time.time())}"
        file_tool.save_json(artifact_id, artifact)
        return {"artifact_id": artifact_id, "artifact": artifact}

agents = {
    "Planner": PlannerAgent("Planner"),
    "ContentGen": ContentGenAgent("ContentGen"),
    "QuizMaster": QuizMasterAgent("QuizMaster"),
    "Evaluator": EvaluatorAgent("Evaluator"),
}

orchestrator = OrchestratorAgent("Orchestrator", agents)

def demo():
    session_id = sessions.create_session("demo_user")
    result = orchestrator.handle_user_request(session_id, {"topic": "Linear Regression", "hours": 4})
    return result

if __name__ == "__main__":
    print(demo())

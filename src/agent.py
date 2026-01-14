from typing import Annotated, Literal, TypedDict, Dict, Any, List
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.rag import get_retriever

load_dotenv()

# --- Config & Tools ---

# --- Config & Tools ---

# Real Gemini Implementation
# Ensure you have GOOGLE_API_KEY in .env
try:
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
except Exception as e:
    # Fallback/Error handling if key is missing, though main.py handles it.
    print(f"Warning: Failed to init Gemini: {e}")
    llm = None

def mock_lead_capture(name: str, email: str, platform: str):
    """Mocks sending lead data to a backend."""
    print(f"DEBUG: Executing mock_lead_capture with {name}, {email}, {platform}")
    return f"Lead captured successfully: {name}, {email}, {platform}"

# --- State Application ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    lead_info: Dict[str, str] # keys: name, email, platform

# --- Nodes ---

def detect_intent(state: AgentState):
    """Analyzes the conversation to determine the user's intent."""
    messages = state["messages"]
    
    # If we are already collecting lead info, we might want to stay in that flow
    # or check if the user is cancelling. But for simplicity, we'll re-evaluate
    # or checking if the user provided info.
    # Actually, a better approach for slot filling is to let the 'manage_lead' node handle extraction.
    # But we need to enter that node first.
    
    system_prompt = """You are an intent classifier for AutoStream.
    Classify the user's latest message into one of these categories:
    1. 'greeting': Casual hello/greeting.
    2. 'inquiry': Questions about pricing, features, policies, or the product.
    3. 'high_intent': User wants to buy, sign up, or try the Pro plan. OR User is providing personal details (name, email) in response to a request.
    
    Return ONLY the category name.
    """
    
    print("DEBUG: detect_intent calling LLM...")
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages[-3:]) # Look at last few messages
    print("DEBUG: detect_intent LLM responded")
    intent = response.content.strip().lower()
    
    # Normalization fallback
    if "greeting" in intent: intent = "greeting"
    elif "inquiry" in intent: intent = "inquiry"
    elif "high_intent" in intent or "sign up" in intent or "details" in intent: intent = "high_intent"
    else: intent = "inquiry" # Default fallback
    
    print(f"DEBUG: Detected intent: {intent}")
    return {"intent": intent}

def respond_greeting(state: AgentState):
    return {"messages": [AIMessage(content="Hi there! I'm the AutoStream assistant. How can I help you today? Check out our pricing or ask about features.")]}

def retrieve_and_respond(state: AgentState):
    """RAG Flow"""
    query = state["messages"][-1].content
    print(f"DEBUG: retrieve_and_respond called with query: {query}")
    # retriever = get_retriever()
    print("DEBUG: Retriever obtained, invoking...")
    # docs = retriever.invoke(query)
    # print(f"DEBUG: Retrieved {len(docs)} docs")
    # context = "\n\n".join([d.page_content for d in docs])
    context = "Pricing: Basic Plan $29/mo, Pro Plan $79/mo."
    
    prompt = f"""You are a helpful assistant for AutoStream. 
    Answer the user's question based on the following context.
    
    Context:
    {context}
    
    Question: {query}
    """
    print("DEBUG: Invoking LLM for answer...")
    response = llm.invoke(prompt)
    print("DEBUG: LLM responded")
    return {"messages": [response]}

def manage_lead(state: AgentState):
    """Handles lead capture: Extraction -> Check Missing -> Ask/Call Tool"""
    messages = state["messages"]
    current_info = state.get("lead_info", {}) or {}
    
    # 1. Extraction
    # We ask the LLM to extract any new info from the last message(s)
    extract_prompt = f"""Current lead info: {current_info}
    User message: {messages[-1].content}
    
    Extract 'name', 'email', and 'platform' from the message if present.
    Return a JSON dict with the found keys. If not found, ignore. 
    Do not hallucinate.
    Example: {{"name": "John"}}
    """
    # For robust extraction in production, use structured output or tools. 
    # Here we'll do a simple text parse or trust the LLM's json mode if enabled, 
    # but flash handles plain text well.
    # Let's try to update current_info.
    
    try:
        extraction = llm.invoke(extract_prompt).content
        import json
        # basic cleanup for JSON
        extraction = extraction.replace("```json", "").replace("```", "").strip()
        if "{" in extraction:
            new_data = json.loads(extraction)
            current_info.update(new_data)
    except:
        pass # extraction failed or empty
    
    # 2. Check completeness
    required = ["name", "email", "platform"]
    missing = [field for field in required if field not in current_info]
    
    if not missing:
        # All present -> Call tool
        result = mock_lead_capture(current_info["name"], current_info["email"], current_info["platform"])
        response_msg = f"Thanks {current_info['name']}! {result}. We'll be in touch."
        # Reset info if needed, or just end
        return {"messages": [AIMessage(content=response_msg)], "lead_info": {}} # Resetting after capture
    else:
        # Ask for missing
        response_msg = f"Great! To get you started, I need a few details. Please provide your {', '.join(missing)}."
        return {"messages": [AIMessage(content=response_msg)], "lead_info": current_info}

# --- Routing ---

def route_intent(state: AgentState):
    intent = state["intent"]
    if intent == "greeting":
        return "greeting"
    elif intent == "inquiry":
        return "rag"
    elif intent == "high_intent":
        return "lead"
    return "rag"

# --- Graph ---

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("respond_greeting", respond_greeting)
    workflow.add_node("retrieve_and_respond", retrieve_and_respond)
    workflow.add_node("manage_lead", manage_lead)
    
    workflow.set_entry_point("detect_intent")
    
    workflow.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            "greeting": "respond_greeting",
            "rag": "retrieve_and_respond",
            "lead": "manage_lead"
        }
    )
    
    workflow.add_edge("respond_greeting", END)
    workflow.add_edge("retrieve_and_respond", END)
    workflow.add_edge("manage_lead", END)
    
    return workflow.compile(checkpointer=MemorySaver())

agent_app = build_graph()


import uuid
from src.agent import agent_app
from langchain_core.messages import HumanMessage

def run_test():
    print("Starting Verification...")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 1. Greeting & Pricing
    print("\n--- Turn 1: Pricing Inquiry ---")
    input1 = "Hi, tell me about your pricing."
    print(f"User: {input1}")
    res1 = agent_app.invoke({"messages": [HumanMessage(content=input1)]}, config=config)
    print(f"Agent: {res1['messages'][-1].content}")
    
    # 2. High Intent
    print("\n--- Turn 2: Buy Pro Plan ---")
    input2 = "That sounds good, I want to try the Pro plan for my YouTube channel."
    print(f"User: {input2}")
    res2 = agent_app.invoke({"messages": [HumanMessage(content=input2)]}, config=config)
    print(f"Agent: {res2['messages'][-1].content}")
    
    # 3. Provide Name
    print("\n--- Turn 3: Provide Name ---")
    input3 = "My name is John Doe."
    print(f"User: {input3}")
    res3 = agent_app.invoke({"messages": [HumanMessage(content=input3)]}, config=config)
    print(f"Agent: {res3['messages'][-1].content}")
    
    # 4. Provide Email (assuming platform might have been picked up or asked for)
    print("\n--- Turn 4: Provide Email ---")
    input4 = "john.doe@example.com"
    print(f"User: {input4}")
    res4 = agent_app.invoke({"messages": [HumanMessage(content=input4)]}, config=config)
    print(f"Agent: {res4['messages'][-1].content}")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Verification Failed: {e}")

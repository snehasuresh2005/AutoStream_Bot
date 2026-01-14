import sys
import uuid
from src.agent import agent_app
from langchain_core.messages import HumanMessage

def main():
    print("Initialize AutoStream Agent...")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    # Unique thread ID for state memory
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Session ID: {thread_id}")
    print("--------------------------------------------------")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
                
            # Run the agent
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # Use stream to show intermediate steps if needed, but invoke is fine for simple CLI
            print("DEBUG: Invoking agent_app...")
            result = agent_app.invoke(inputs, config=config)
            print("DEBUG: agent_app returned")
            
            # Get the last AI message
            last_message = result["messages"][-1]
            print(f"Agent: {last_message.content}")
            print("--------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

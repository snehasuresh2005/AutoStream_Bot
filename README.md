# AutoStream Lead Agent
A RAG-powered Conversational AI Agent for qualifying business leads.

## 1. How to Run Locally

### Prerequisites
- Python 3.10+
- A Google Gemini API Key

### Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   - Create a `.env` file in the root.
   - Add your key: `GOOGLE_API_KEY=your_key_here`
4. **Run the Agent**:
   ```bash
   python main.py
   ```
   *Note: Ensure your `GOOGLE_API_KEY` has access to the `gemini-flash-latest` model.*

## 2. Architecture Explanation

The AutoStream Bot is architected using **LangGraph**, a framework designed for building stateful, multi-actor applications with LLMs. 

### Why LangGraph?
We selected LangGraph over alternatives like AutoGen or purely linear chains for several key reasons:
1.  **State Management**: It allows us to explicitly define an `AgentState` schema (a typed dictionary) that persists conversation history, user intent, and captured lead data across the lifecycle of the interaction.
2.  **Cyclic Control Flow**: Conversational AI is rarely linear. LangGraph supports cyclic response loops (e.g., the bot repeatedly asking for missing lead details until satisfied) which are difficult to model in DAG-based frameworks.
3.  **Determinism & Guardrails**: By using strict "Conditional Edges" (Routing), we ensure the bot transitions predictably between modes (e.g., from `Greeting` to `RAG` or `Lead Capture`) based on classified intent, preventing the "hallucination loop" common in fully autonomous agents.

### Core Components
- **State**: Tracks `messages` (chat history) and `lead_info` (e.g., `{"name": "...", "email": "..."}`).
- **Nodes**:
    - `detect_intent`: The entry point that analyzes user input using the LLM to route the flow.
    - `retrieve_and_respond`: A RAG node that embeds the query, searches the ChromaDB vector store, and synthesizes an answer from the retrieved policy documents.
    - `manage_lead`: A specialized node for slot-filling. It checks for missing fields (Name, Email) and loops back to the user if data is incomplete, or triggers the `mock_lead_capture` tool once finished.
- **Memory**: We employ `MemorySaver` to checkpoint the state after every step, allowing for long-running conversations that can resume seamlessly.

## 3. WhatsApp Deployment Strategy
To integrate this agent with WhatsApp, you would interact with the **WhatsApp Business API** (via Meta) using Webhooks.

### Integration Steps
1. **Deploy a Webhook Server**: 
   Deploy a lightweight backend service (using FastAPI or Flask) exposing a `POST /webhook` endpoint.
   
2. **Verify the Webhook**: 
   Configure the URL in the Meta Developer Portal. Your server must handle the `GET` verification challenge (`hub.verify_token`) to confirm ownership.

3. **Handle Incoming Messages**:
   - When a user messages your WhatsApp number, Meta sends a JSON payload to your server.
   - Extract the user's `phone_number` and the message `text.body`.

4. **Connect to Agent (State Management)**:
   - Use the extracted `phone_number` as the **`thread_id`** in the LangGraph configuration.
   - This ensures the agent maintains a unique, persistent conversation state for each specific user automatically.
   ```python
   # Example integration snippet
   config = {"configurable": {"thread_id": user_phone_number}}
   response = agent_app.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
   ```

5. **Send Reply**:
   - Take the agent's final text response.
   - Make an HTTP `POST` request to the WhatsApp Graph API (`/messages` endpoint) to send the text back to the user.

## Features
- **Intent Detection**: Classifies users into Greeting, Inquiry (Product/Pricing), or High Intent.
- **RAG Pipeline**: Retrieves knowledge from `data/knowledge_base.md` to answer questions.
- **Lead Capture**: Interactive slot-filling state machine to collect Name, Email, and Platform before "capturing" the lead.

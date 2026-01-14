# AutoStream Social-to-Lead Agent

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

1. **Backend Service**: Create a backend (using FastAPI or Flask) to expose a Webhook URL (e.g., `POST /webhook`).
2. **Webhook Configuration**: Register this URL in the Meta Developer Portal. Meta validates the URL with a `verify_token` GET request.
3. **Message Handling**:
   - When a user messages the WhatsApp number, Meta sends a JSON POST payload to the webhook.
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

   - The payload contains the user's `phone_number` and `text.body`.
   - We use the `phone_number` as the `thread_id` in LangGraph to maintain per-user state.
4. **Agent Invocation**: Pass the text to `agent_app.invoke({"messages": [...]}, config={"configurable": {"thread_id": phone_number}})`.
5. **Response**: The agent's text response is sent back to the user via the `POST /messages` endpoint of the WhatsApp API.

This architecture ensures scalable, stateful conversations where the specialized "Agent" logic remains decoupled from the messaging channel transport layer.

## Troubleshooting

### ModuleNotFoundError: No module named 'langgraph'
If you see this error, it means the dependencies failed to install, likely due to **SSL/Network errors** (common in corporate environments).

**Fix 1: Trust PyPI Hosts**
Run this command to bypass SSL verification for package servers:
```bash
.\venv\bin\python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**Fix 2: Install Pre-built Wheels (Avoid Build Errors)**
Try installing only binary wheels to avoid compiling `numpy` from source:
```bash
.\venv\bin\python -m pip install --only-binary=:all: -r requirements.txt
```

**Fix 3: Manual Install**
If all else fails, try installing the key packages individually:
```bash
.\venv\bin\python -m pip install langgraph langchain langchain-google-genai chromadb
```


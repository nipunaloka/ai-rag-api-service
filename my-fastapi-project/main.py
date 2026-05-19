import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx

# LangChain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# MongoDB
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
MONGO_URI           = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB            = os.getenv("MONGO_DB", "homelanka")
MONGO_COLLECTION    = os.getenv("MONGO_COLLECTION", "properties")
PROPERTY_API_URL    = os.getenv("PROPERTY_API_URL", "https://homelanka.lk/listings")   
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="HomeLanka RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ────────────────────────────────────────────────────
session_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# ── LLM & Embeddings ──────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    temperature=0,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_docs_from_mongodb() -> list[Document]:
    """
    Reads property documents from MongoDB and converts each record
    into a LangChain Document with metadata.
    """
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]

        documents = []
        for prop in collection.find():
            # Build a natural-language description from the MongoDB fields.
            # Adjust field names to match YOUR actual MongoDB schema.
            content = (
                f"{prop.get('bedrooms', '?')} bedroom {prop.get('type', 'property')} "
                f"in {prop.get('location', 'Unknown')}, "
                f"rent {prop.get('price', '?')} LKR, "
                f"{prop.get('description', '')}"
            )
            metadata = {
                "property_id": str(prop.get("_id", "")),
                "location": prop.get("location", ""),
                "price": prop.get("price", 0),
                "type": prop.get("type", ""),
            }
            documents.append(Document(page_content=content, metadata=metadata))

        client.close()
        print(f"[MongoDB] Loaded {len(documents)} property documents.")
        return documents

    except Exception as e:
        print(f"[MongoDB] Failed to connect: {e}. Falling back to static docs.")
        return _static_fallback_docs()


async def load_docs_from_property_api() -> list[Document]:
    """
    Optional: Fetches fresh property listings from your existing backend API
    and converts them to LangChain Documents.
    Only used when PROPERTY_API_URL is set in .env
    """
    if not PROPERTY_API_URL:
        return []

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(PROPERTY_API_URL)
            resp.raise_for_status()
            properties = resp.json()  # expects a list of property dicts

        documents = []
        for prop in properties:
            content = (
                f"{prop.get('bedrooms', '?')} bedroom {prop.get('type', 'property')} "
                f"in {prop.get('location', 'Unknown')}, "
                f"rent {prop.get('price', '?')} LKR, "
                f"{prop.get('description', '')}"
            )
            documents.append(Document(page_content=content, metadata=prop))

        print(f"[Property API] Loaded {len(documents)} properties.")
        return documents

    except Exception as e:
        print(f"[Property API] Error: {e}")
        return []


def _static_fallback_docs() -> list[Document]:
    """Fallback static documents (same as your notebook)."""
    return [
        Document(page_content="3 bedroom house in Colombo 05, rent 80000 LKR, near school and supermarket"),
        Document(page_content="2 bedroom apartment in Kandy city, rent 50000 LKR, mountain view"),
        Document(page_content="Luxury villa in Galle with sea view, rent 150000 LKR, private pool"),
        Document(page_content="Small house in Negombo, rent 40000 LKR, near beach and airport"),
        Document(page_content="Modern apartment in Colombo 03, rent 90000 LKR, city center location"),
        Document(page_content="""
HomeLanka.lk is a digital real estate marketplace platform in Sri Lanka.
It connects property buyers, sellers, and renters.
Users can browse listings, search properties, and find homes based on location, price, and preferences.
The platform provides verified property listings and an easy-to-use search system.
"""),
    ]


# ── Build RAG chain ────────────────────────────────────────────────────────────

def build_rag_chain(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # History-aware retriever prompt
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant for HomeLanka, a real estate platform in Sri Lanka.
Reformulate the user's latest question into a standalone search query.
Keep property intent (price, location, type). Do NOT answer — only rewrite the query."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Answer prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are HomeLanka AI, a smart real estate assistant for Sri Lanka.

Your job is to help users with:
1. Property search (houses, apartments, villas)
2. Information about HomeLanka platform

RULES:
- If context contains relevant property information → use it to answer
- If question is about HomeLanka platform → answer using context or general knowledge
- Do NOT invent property listings or fake data
- Be clear, helpful, and concise

When answering property questions:
- Show best matching properties
- Include price, location, and key features
- Give a short recommendation

Context:
{context}
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "User question: {input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_chain


# ── App startup ───────────────────────────────────────────────────────────────

conversational_rag_chain = None   # initialised in startup

@app.on_event("startup")
async def startup_event():
    global conversational_rag_chain

    # 1. Try MongoDB first
    docs = load_docs_from_mongodb()

    # 2. If you also want to merge data from your existing backend API, uncomment:
    # api_docs = await load_docs_from_property_api()
    # docs = docs + api_docs

    # 3. Fall back to static if nothing loaded
    if not docs:
        docs = _static_fallback_docs()

    conversational_rag_chain = build_rag_chain(docs)
    print("[Startup] RAG chain is ready.")


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"   # use different IDs per user/conversation

class ChatResponse(BaseModel):
    answer: str
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "HomeLanka RAG API is running. POST /chat to query."}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Postman: POST http://localhost:8000/chat
    Body (JSON):
        { "message": "What houses are available in Colombo?", "session_id": "user_123" }
    """
    if conversational_rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not ready yet.")

    response = conversational_rag_chain.invoke(
        {"input": req.message},
        config={"configurable": {"session_id": req.session_id}},
    )
    return ChatResponse(answer=response["answer"], session_id=req.session_id)


@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    """Clear chat history for a specific session."""
    if session_id in session_store:
        del session_store[session_id]
    return {"message": f"Session '{session_id}' cleared."}


@app.post("/reload")
async def reload_documents():
    """
    Re-fetch documents from MongoDB (and optionally the property API)
    and rebuild the vector store. Call this when new properties are added.
    """
    global conversational_rag_chain
    docs = load_docs_from_mongodb()
    if not docs:
        docs = _static_fallback_docs()
    conversational_rag_chain = build_rag_chain(docs)
    return {"message": f"Reloaded {len(docs)} documents and rebuilt RAG chain."}
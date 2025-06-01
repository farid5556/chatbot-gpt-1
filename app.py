from fastapi import FastAPI, Request
from pydantic import BaseModel
from retriever import get_vectorstore
from generator import generate_answer

app = FastAPI()
vectorstore = get_vectorstore()

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(question: Question):
    docs = vectorstore.similarity_search(question.query, k=2)
    context = " ".join([doc.page_content for doc in docs])
    answer = generate_answer(context, question.query)
    return {"answer": answer}


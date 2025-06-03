from retriever import get_vectorstore
from generator import generate_answer

vectorstore = get_vectorstore()

print("🤖 Chatbot RAG siap! Ketik 'exit' untuk keluar.")
while True:
    question = input("🧠 Pertanyaanmu: ")
    if question.lower() == "exit":
        break

    # Ambil konteks dari Weaviate
    docs = vectorstore.similarity_search(question, k=2)
    context = " ".join([doc.page_content for doc in docs])

    # Dapatkan jawaban dari model T5
    answer = generate_answer(context, question)
    print(f"💬 Jawaban: {answer}\n")

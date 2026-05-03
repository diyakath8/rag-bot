from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

DB_PATH = "vectorstore"

def main():
    print("Loading vector DB...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7, "fetch_k": 20}
    )

    llm = OllamaLLM(model="llama3")

    print("💬 Ask questions (type 'exit' to quit)")

    while True:
        query = input("\nYou: ")

        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.

If the context clearly contains relevant information, provide a complete answer.

ONLY say "I don't know" if the context is completely unrelated to the question.

Do NOT add "I don't know" if you already gave a valid answer.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        print("\nAnswer:\n")
        print(response)

        print("\nSources:")
        for doc in docs:
            print(doc.metadata)


if __name__ == "__main__":
    main()
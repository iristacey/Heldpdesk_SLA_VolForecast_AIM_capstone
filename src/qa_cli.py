from src.qa_assistant import answer_question

if __name__ == "__main__":
    q = input("Ask a project question: ")
    print(answer_question(q))

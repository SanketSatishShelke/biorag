# ui/cli.py
"""
BioRAG terminal client — thin HTTP client against the /query endpoint.
REPL-style: stays open, holds conversation history in memory, folds prior
turns into follow-up questions before sending to the API.
"""
import os
import requests

API_URL = os.getenv("BIORAG_API_URL", "http://127.0.0.1:8000")
NAMESPACE = os.getenv("BIORAG_NAMESPACE", "default")
TOP_K = int(os.getenv("BIORAG_TOP_K", "5"))

MAX_HISTORY_TURNS = 3        # how many prior turns to fold into context
ANSWER_TRUNCATE = 500        # keep folded-in prior answers short


def build_query(history: list[tuple[str, str]], new_question: str) -> str:
    """Fold recent history into the question sent for retrieval rewriting."""
    if not history:
        return new_question

    context_lines = []
    for q, a in history[-MAX_HISTORY_TURNS:]:
        a_short = a if len(a) <= ANSWER_TRUNCATE else a[:ANSWER_TRUNCATE] + "..."
        context_lines.append(f"Previous question: {q}\nPrevious answer: {a_short}")

    context_block = "\n\n".join(context_lines)
    return f"{context_block}\n\nFollow-up question: {new_question}"


def call_query(question: str) -> dict:
    resp = requests.post(
        f"{API_URL}/query",
        data={"question": question, "namespace": NAMESPACE, "top_k": TOP_K},
        timeout=120,
    )
    if resp.status_code == 400:
        detail = resp.json().get("detail", {})
        if isinstance(detail, dict):
            raise ValueError(f"[{detail.get('error', 'REJECTED')}] {detail.get('message', 'Request rejected')}")
        raise ValueError(str(detail))
    resp.raise_for_status()
    return resp.json()


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n")

    if result.get("confidence") == "LOW":
        print("  (confidence: LOW)")

    sources = result.get("sources", [])
    if sources:
        print("Sources:")
        for s in sources:
            print(f"  - {s['filename']} (p.{s['page_number']}, score={s['score']})")
    print(f"\n[retrieved {result['chunks_retrieved']} chunks | rewritten query: \"{result['retrieval_query']}\"]")


def main():
    print("BioRAG terminal client")
    print(f"API: {API_URL} | namespace: {NAMESPACE} | top_k: {TOP_K}")
    print("Type your question, 'reset' to clear conversation history, or 'exit' to quit.\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            question = input("biorag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            history.clear()
            print("Conversation history cleared.\n")
            continue

        query_text = build_query(history, question)

        try:
            result = call_query(query_text)
        except ValueError as e:
            print(f"\nRejected: {e}\n")
            continue
        except requests.exceptions.ConnectionError:
            print(f"\nCould not reach BioRAG API at {API_URL}. Is it running?\n")
            continue
        except requests.exceptions.HTTPError as e:
            print(f"\nAPI error: {e}\n")
            continue

        print_result(result)
        history.append((question, result["answer"]))
        print()


if __name__ == "__main__":
    main()

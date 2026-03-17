from utils.logger import log
import ast


def grade_documents(state):
    """
    Grade retrieved documents for relevance to the question.

    Scoring logic:
    - Extract question keywords (words > 3 chars to skip stop words)
    - Count how many keywords appear in the retrieved context
    - If >= 2 keywords match → relevant
    - If < 2 keywords match → not relevant → triggers rewrite

    This replaces the old no-op grader that always passed everything.
    """

    log("Grading retrieved documents...")

    messages = state["messages"]
    rewrite_count = state.get("rewrite_count", 0)

    # First message is always the user question
    question = messages[0].content.lower()

    # Last message is the tool (retriever) output
    raw_context = messages[-1].content

    # Parse the retriever dict to get clean text
    try:
        parsed = ast.literal_eval(raw_context)
        context = parsed.get("text", raw_context).lower()
    except Exception:
        context = raw_context.lower()

    # Extract meaningful keywords — skip short stop words
    keywords = [w for w in question.split() if len(w) > 3]
    matched = sum(1 for kw in keywords if kw in context)
    total = len(keywords) if keywords else 1
    relevance_ratio = matched / total

    log(f"Keywords matched: {matched}/{total} (ratio: {relevance_ratio:.2f})")

    # Relevance threshold — needs at least 2 keyword hits OR 40% keyword coverage
    if matched >= 2 or relevance_ratio >= 0.4:
        log("✅ Documents graded: RELEVANT — proceeding to answer")
        return {
            "messages": messages,
            "doc_grade": "relevant",
            "rewrite_count": rewrite_count
        }
    else:
        # Only rewrite once — after 1 retry, accept whatever we have
        if rewrite_count >= 1:
            log("⚠️ Documents still weak after rewrite — proceeding anyway")
            return {
                "messages": messages,
                "doc_grade": "relevant",
                "rewrite_count": rewrite_count
            }
        else:
            log("❌ Documents graded: NOT RELEVANT — rewriting question")
            return {
                "messages": messages,
                "doc_grade": "not_relevant",
                "rewrite_count": rewrite_count
            }
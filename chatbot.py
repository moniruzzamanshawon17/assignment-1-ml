"""
chatbot.py
----------
Builds the full LCEL pipeline:

    question -> classifier -> RunnableBranch -> RunnableParallel -> ChatResponse

Satisfies FR-4 (RunnableBranch) and FR-5 (RunnableParallel).
"""

import os

from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq

from prompts import (
    CLASSIFIER_PROMPT,
    GENERAL_PROMPT,
    INSIGHTS_PROMPT,
    MATH_PROMPT,
    PROGRAMMING_PROMPT,
)
from schemas import Answer, Category, ChatResponse, Insights, QuestionRoute

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_llm(temperature: float = 0.3) -> ChatGroq:
    """Create the Groq chat model. Key is read from .env, never hardcoded."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Copy .env.example to .env and add your key."
        )
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        api_key=api_key,
    )


# The dict flowing through the chain carries {question, category, reason}, but
# the prompts only need {question}. Trimming it here keeps each prompt's input
# contract explicit and stops routing metadata leaking into the prompt text.
only_question = RunnableLambda(lambda x: {"question": x["question"]})


def merge_results(data: dict) -> ChatResponse:
    """Combine the two parallel branches and the routing info into one object."""
    answer: Answer = data["answer"]
    insights: Insights = data["insights"]
    route: dict = data["route"]

    return ChatResponse(
        category=route["category"],
        routing_reason=route["reason"],
        answer=answer.answer,
        confidence=answer.confidence,
        summary=insights.summary,
        keywords=insights.keywords,
        difficulty=insights.difficulty,
        follow_up_questions=insights.follow_up_questions,
    )


def build_chatbot():
    """Assemble and return the complete chain."""
    llm = get_llm()

    # --- Step 1: classify the question so the branch has something to test ---
    classifier_chain = CLASSIFIER_PROMPT | llm.with_structured_output(QuestionRoute)

    classify_stage = RunnableParallel(
        question=RunnableLambda(lambda x: x["question"]),
        route=classifier_chain,
    ) | RunnableLambda(
        lambda x: {
            "question": x["question"],
            "category": x["route"].category,
            "reason": x["route"].reason,
        }
    )

    # --- Step 2: three specialist chains, one per route (FR-4) ---
    programming_chain = only_question | PROGRAMMING_PROMPT | llm.with_structured_output(Answer)
    math_chain = only_question | MATH_PROMPT | llm.with_structured_output(Answer)
    general_chain = only_question | GENERAL_PROMPT | llm.with_structured_output(Answer)

    answer_branch = RunnableBranch(
        (lambda x: x["category"] == Category.PROGRAMMING, programming_chain),
        (lambda x: x["category"] == Category.MATH, math_chain),
        general_chain,  # default route
    )

    # --- Step 3: run the answer and the analysis at the same time (FR-5) ---
    insights_chain = only_question | INSIGHTS_PROMPT | llm.with_structured_output(Insights)

    parallel_stage = RunnableParallel(
        answer=answer_branch,
        insights=insights_chain,
        route=RunnablePassthrough(),
    )

    # --- Step 4: merge into a single validated ChatResponse (FR-3) ---
    return classify_stage | parallel_stage | RunnableLambda(merge_results)


def ask(chain, question: str) -> ChatResponse:
    """Run one question through the pipeline."""
    return chain.invoke({"question": question})
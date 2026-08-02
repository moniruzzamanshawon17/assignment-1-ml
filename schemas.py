from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The three routes handled by RunnableBranch (FR-4)."""

    PROGRAMMING = "programming"
    MATH = "math"
    GENERAL = "general"


class Difficulty(str, Enum):
    """How advanced the question is."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class QuestionRoute(BaseModel):
    """
    Output of the classifier chain.

    This is what makes RunnableBranch possible: the branch needs a concrete
    field to test against, so we ask the LLM to label the question first.
    """

    category: Category = Field(
        description=(
            "Which specialist should handle this question. "
            "Use 'programming' for code, software, algorithms, or debugging. "
            "Use 'math' for calculations, proofs, statistics, or equations. "
            "Use 'general' for everything else."
        )
    )
    reason: str = Field(
        description="One short sentence justifying the chosen category."
    )


class Answer(BaseModel):
    """First branch of RunnableParallel: the substantive reply."""

    answer: str = Field(
        description="A complete, well-explained answer to the user's question."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident you are in this answer, from 0.0 to 1.0.",
    )


class Insights(BaseModel):
    """Second branch of RunnableParallel: metadata about the question."""

    summary: str = Field(
        description="A one-sentence summary of what the user is asking."
    )
    keywords: List[str] = Field(
        min_length=3,
        max_length=6,
        description="Three to six topical keywords for this question.",
    )
    difficulty: Difficulty = Field(
        description="The difficulty level of the question."
    )
    follow_up_questions: List[str] = Field(
        min_length=2,
        max_length=3,
        description="Two or three natural follow-up questions the user might ask next.",
    )


class ChatResponse(BaseModel):
    """
    The final object rendered by Streamlit.

    Built by merging the two RunnableParallel branches together with the
    routing decision, so the UI only ever deals with one validated model.
    """

    category: Category
    routing_reason: str
    answer: str
    confidence: float
    summary: str
    keywords: List[str]
    difficulty: Difficulty
    follow_up_questions: List[str]
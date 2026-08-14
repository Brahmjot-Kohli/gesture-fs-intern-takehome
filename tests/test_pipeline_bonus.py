"""Fast unit tests for pipeline behavior that does not require real AI models."""

from types import SimpleNamespace

from src.pipeline import PROMPT_TEMPLATE, ask_question


class FakeVectorStore:
    """Small FAISS substitute that records how it was searched."""

    def __init__(self, source_texts):
        self.documents = [
            SimpleNamespace(page_content=text) for text in source_texts
        ]
        self.search_calls = []

    def similarity_search(self, question, k):
        self.search_calls.append((question, k))
        return self.documents[:k]


class FakeLLM:
    """Small LLM substitute that records prompts and returns fixed text."""

    def __init__(self, generated_text):
        self.generated_text = generated_text
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return [{"generated_text": self.generated_text}]


def test_empty_question_skips_retrieval_and_generation():
    """Whitespace-only questions should be handled without calling dependencies."""

    class DependencyThatMustNotRun:
        def similarity_search(self, question, k):
            raise AssertionError("Vector search should not run for an empty question")

        def __call__(self, prompt):
            raise AssertionError("The LLM should not run for an empty question")

    dependency = DependencyThatMustNotRun()
    result = ask_question(dependency, dependency, "   ")

    assert result == {
        "answer": "Please provide a valid question.",
        "sources": [],
    }


def test_retrieves_three_chunks_and_returns_their_text():
    """The response should expose the same three chunks used as context."""

    source_texts = ["First source", "Second source", "Third source"]
    vector_store = FakeVectorStore(source_texts + ["Unused fourth source"])
    llm = FakeLLM("A generated answer")

    result = ask_question(vector_store, llm, "What services do you offer?")

    assert vector_store.search_calls == [("What services do you offer?", 3)]
    assert result["sources"] == source_texts


def test_formats_the_provided_prompt_with_context_and_question():
    """Retrieved text and the client's question should reach the LLM prompt."""

    question = "How much is the Growth package?"
    source_texts = ["Growth costs $5,500.", "Six-month commitment."]
    vector_store = FakeVectorStore(source_texts)
    llm = FakeLLM("The Growth package costs $5,500.")

    ask_question(vector_store, llm, question)

    expected_prompt = PROMPT_TEMPLATE.format(
        context="\n\n".join(source_texts),
        question=question,
    )
    assert llm.prompts == [expected_prompt]


def test_strips_generated_answer_whitespace():
    """Callers should receive clean answer text from the LLM result."""

    vector_store = FakeVectorStore(["Starter costs $2,500 per month."])
    llm = FakeLLM("  The Starter package costs $2,500 per month.  \n")

    result = ask_question(vector_store, llm, "What does Starter cost?")

    assert result["answer"] == "The Starter package costs $2,500 per month."


def test_empty_generated_text_uses_fallback_answer():
    """An empty model response should still produce a useful answer."""

    vector_store = FakeVectorStore(["Some relevant context"])
    llm = FakeLLM("   ")

    result = ask_question(vector_store, llm, "What can you tell me?")

    assert result["answer"] == "I don't have enough information to answer that."

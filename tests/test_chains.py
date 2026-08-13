"""
Tests for the composition layer: runnables, prompts, parsers, RAG.

Companion to `test_agentkit.py`, which covers the graph runtime. Split into
two files because they test two genuinely different things: this one is about
*chains* (linear, functional, stateless), the other about *graphs* (cyclic,
stateful, resumable).

    python tests/test_chains.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit import (  # noqa: E402
    BooleanOutputParser,
    ChatPromptTemplate,
    Document,
    FakeModel,
    FewShotPromptTemplate,
    HashingEmbeddings,
    InMemoryHistoryStore,
    InMemoryVectorStore,
    JSONOutputParser,
    ListOutputParser,
    Message,
    MessagesPlaceholder,
    OutputParserError,
    PromptTemplate,
    RecursiveTextSplitter,
    RunnableBranch,
    RunnableFallback,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableRetry,
    RunnableWithMessageHistory,
    StrOutputParser,
    StructuredOutputParser,
    create_rag_chain,
    format_documents,
)
from agentkit.errors import ConfigurationError  # noqa: E402


# ===========================================================================
# prompts
# ===========================================================================
def test_template_variables_and_rendering():
    t = PromptTemplate("Write about {topic} for {audience}.")
    assert t.input_variables == ["topic", "audience"]
    assert t.format(topic="RAG", audience="beginners") == "Write about RAG for beginners."


def test_missing_variable_is_a_clear_error():
    t = PromptTemplate("Hello {name}.")
    try:
        t.format()
    except ConfigurationError as e:
        assert "name" in str(e)
    else:
        raise AssertionError("expected ConfigurationError")


def test_escaped_braces_survive():
    """JSON examples inside prompts must not be treated as variables."""
    t = PromptTemplate('Return {{"ok": true}} for {topic}.')
    assert t.input_variables == ["topic"]
    assert t.format(topic="x") == 'Return {"ok": true} for x.'


def test_partial_prefills():
    t = PromptTemplate("{company} asks about {topic}.").partial(company="Acme")
    assert t.input_variables == ["topic"]
    assert t.format(topic="refunds") == "Acme asks about refunds."


def test_chat_template_roles_and_placeholder():
    t = ChatPromptTemplate.from_messages(
        [("system", "You are {persona}."), MessagesPlaceholder("history"), ("user", "{input}")]
    )
    msgs = t.format_messages(
        persona="terse", history=[Message.user("earlier"), Message.assistant("noted")], input="now"
    )
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0].content == "You are terse."
    assert msgs[-1].content == "now"


def test_placeholder_is_optional_when_history_absent():
    t = ChatPromptTemplate.from_messages(
        [("system", "s"), MessagesPlaceholder("history"), ("user", "{input}")]
    )
    assert len(t.format_messages(input="hi")) == 2


def test_few_shot_renders_examples():
    t = FewShotPromptTemplate(
        examples=[{"in": "happy", "out": "sad"}, {"in": "tall", "out": "short"}],
        example_template="{in} -> {out}",
        suffix="{word} ->",
        prefix="Give the opposite.",
    )
    out = t.format(word="fast")
    assert "happy -> sad" in out and "tall -> short" in out and out.strip().endswith("fast ->")


# ===========================================================================
# runnables / LCEL
# ===========================================================================
def test_pipe_composition():
    chain = (
        PromptTemplate("Say hi to {name}.")
        | FakeModel(responses=["Hi Ada."])
        | StrOutputParser()
    )
    assert chain.invoke({"name": "Ada"}) == "Hi Ada."


def test_chain_flattens_not_nests():
    chain = PromptTemplate("{x}") | FakeModel(responses=["a"]) | StrOutputParser()
    assert len(chain.steps) == 3  # not two nested pairs


def test_single_variable_accepts_bare_string():
    chain = PromptTemplate("Echo: {text}") | RunnableLambda(lambda s: s.upper())
    assert chain.invoke("hello") == "ECHO: HELLO"


def test_parallel_branches_share_input():
    p = RunnableParallel(
        {"upper": lambda s: s.upper(), "length": lambda s: len(s), "same": RunnablePassthrough()}
    )
    assert p.invoke("abc") == {"upper": "ABC", "length": 3, "same": "abc"}


def test_passthrough_assign_extends_dict():
    r = RunnablePassthrough.assign(words=lambda d: len(d["text"].split()))
    assert r.invoke({"text": "one two three"}) == {"text": "one two three", "words": 3}


def test_batch_preserves_order():
    chain = RunnableLambda(lambda x: x * 2)
    assert chain.batch([1, 2, 3, 4, 5]) == [2, 4, 6, 8, 10]


def test_branch_routes_first_match():
    b = RunnableBranch(
        (lambda x: x < 0, lambda x: "negative"),
        (lambda x: x == 0, lambda x: "zero"),
        lambda x: "positive",
    )
    assert [b.invoke(v) for v in (-1, 0, 7)] == ["negative", "zero", "positive"]


def test_retry_then_succeed():
    calls = {"n": 0}

    def flaky(x, config=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert RunnableRetry(flaky, attempts=3, delay=0.001).invoke(None) == "ok"


def test_fallback_uses_second_provider():
    def broken(x, config=None):
        raise RuntimeError("provider down")

    chain = RunnableFallback(broken, [lambda x: "served by backup"])
    assert chain.invoke("q") == "served by backup"


def test_helpful_error_when_prompt_stage_is_missing():
    chain = RunnableLambda(lambda x: {"oops": x}) | FakeModel(responses=["never reached"])
    try:
        chain.invoke("hi")
    except ConfigurationError as e:
        assert "forget a prompt" in str(e)
    else:
        raise AssertionError("expected a ConfigurationError naming the missing prompt")


# ===========================================================================
# memory
# ===========================================================================
def test_history_accumulates_and_is_session_scoped():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "s"), MessagesPlaceholder("history"), ("user", "{input}")]
    )
    model = FakeModel(responses=["r1", "r2", "r3"])
    store = InMemoryHistoryStore()
    bot = RunnableWithMessageHistory(prompt | model, store.get)

    bot.invoke("first", {"session_id": "a"})
    bot.invoke("second", {"session_id": "a"})
    bot.invoke("other user", {"session_id": "b"})

    assert len(store.get("a").messages) == 4  # 2 user + 2 assistant
    assert len(store.get("b").messages) == 2

    # The second call must have SEEN the first exchange.
    second_call = model.calls[1]
    assert any(m.content == "first" for m in second_call)


# ===========================================================================
# parsers
# ===========================================================================
def test_json_parser_strips_fences_and_prose():
    p = JSONOutputParser()
    assert p.parse('{"a": 1}') == {"a": 1}
    assert p.parse('```json\n{"a": 1}\n```') == {"a": 1}
    assert p.parse('Sure! {"a": 1} Hope that helps.') == {"a": 1}


def test_json_parser_handles_braces_inside_strings():
    """The naive first-brace-to-last-brace approach breaks on this."""
    p = JSONOutputParser()
    assert p.parse('Here: {"note": "use {curly} braces"} done') == {"note": "use {curly} braces"}


def test_json_parser_raises_with_raw_text():
    try:
        JSONOutputParser().parse("no json at all here")
    except OutputParserError as e:
        assert "no json at all" in e.raw
    else:
        raise AssertionError("expected OutputParserError")


def test_structured_parser_validates_fields():
    p = StructuredOutputParser({"a": "an int", "b": "a string"})
    assert p.parse('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}

    for bad in ('{"a": 1}', '{"a": 1, "b": "x", "c": 3}'):
        try:
            p.parse(bad)
        except OutputParserError:
            pass
        else:
            raise AssertionError(f"expected rejection of {bad}")


def test_structured_parser_format_instructions_mention_every_field():
    p = StructuredOutputParser({"sentiment": "positive or negative", "score": "0 to 1"})
    text = p.get_format_instructions()
    assert "sentiment" in text and "score" in text


def test_list_parser_handles_three_shapes():
    p = ListOutputParser()
    assert p.parse('["a", "b"]') == ["a", "b"]
    assert p.parse("- a\n- b\n- c") == ["a", "b", "c"]
    assert p.parse("1. a\n2. b") == ["a", "b"]
    assert p.parse("a, b, c") == ["a", "b", "c"]


def test_boolean_parser_ignores_trailing_explanation():
    p = BooleanOutputParser()
    assert p.parse("Yes, that is correct because...") is True
    assert p.parse("no") is False


# ===========================================================================
# rag
# ===========================================================================
def test_splitter_respects_size_and_overlaps():
    text = "\n\n".join(f"Paragraph {i} with several words in it." for i in range(12))
    chunks = RecursiveTextSplitter(chunk_size=120, chunk_overlap=20).split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 160 for c in chunks)  # size + overlap slack


def test_splitter_drops_contentless_fragments():
    chunks = RecursiveTextSplitter(chunk_size=30, chunk_overlap=5).split_text(
        "Alpha beta gamma. Delta epsilon zeta. Eta theta."
    )
    assert all(any(ch.isalnum() for ch in c) for c in chunks)


def test_splitter_preserves_metadata():
    docs = RecursiveTextSplitter(chunk_size=40, chunk_overlap=5).split_documents(
        [Document("word " * 40, {"source": "a.md"})]
    )
    assert len(docs) > 1
    assert all(d.metadata["source"] == "a.md" for d in docs)
    assert docs[0].metadata["chunk_index"] == 0


def test_embeddings_shared_terms_score_above_unrelated():
    """Regression guard: signed hashing used to let a collision cancel a real
    match exactly, scoring an obviously relevant document at 0.0."""
    store = InMemoryVectorStore(HashingEmbeddings(dimensions=256))
    store.add_texts(
        [
            "Refunds are accepted within 30 days of delivery for damaged items.",
            "Deployment runs through continuous integration on merge.",
        ]
    )
    hits = store.similarity_search("refund for a damaged delivery", k=2)
    assert hits[0].score > 0
    assert "Refunds" in hits[0].content


def test_score_threshold_filters():
    store = InMemoryVectorStore()
    store.add_texts(["alpha beta gamma", "completely unrelated content here"])
    assert len(store.similarity_search("alpha beta", k=5, score_threshold=0.99)) == 0


def test_metadata_filter_applies_before_scoring():
    """The multi-tenancy hook: a filtered-out doc must never be returned."""
    store = InMemoryVectorStore()
    store.add_texts(
        ["tenant one secret data", "tenant two secret data"],
        [{"tenant": "one"}, {"tenant": "two"}],
    )
    hits = store.similarity_search("secret data", k=5, filter={"tenant": "one"})
    assert len(hits) == 1 and hits[0].metadata["tenant"] == "one"


def test_format_documents_numbers_and_labels_sources():
    text = format_documents([Document("body text", {"source": "a.md"})])
    assert "[1]" in text and "a.md" in text


def test_format_documents_when_nothing_retrieved():
    assert "No relevant documents" in format_documents([])


def test_rag_chain_end_to_end():
    store = InMemoryVectorStore()
    store.add_texts(
        ["The refund window is 30 days from delivery."], [{"source": "policy.md"}]
    )
    model = FakeModel(responses=["The refund window is 30 days [1]."])
    chain = create_rag_chain(model, store.as_retriever(k=1))

    answer = chain.invoke("How long is the refund window?")
    assert "30 days" in answer

    # The model must actually have been given the retrieved context, not just
    # the bare question - this is the assertion that catches a broken chain.
    system_message = model.calls[0][0]
    assert "30 days from delivery" in system_message.content


# ===========================================================================
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

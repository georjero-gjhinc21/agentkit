"""
Example 06 — the company chatbot: LCEL + memory + RAG.

This is the build from the LangChain walkthrough, done with this framework:
a chatbot that remembers the conversation, answers from a company knowledge
base, and can swap model providers with a one-line change.

It runs offline. `FakeModel` and `HashingEmbeddings` stand in for a real
provider; the pipeline around them is the real thing.

    python examples/06_rag_chatbot.py
"""

from agentkit import (
    ChatPromptTemplate,
    Document,
    FakeModel,
    InMemoryHistoryStore,
    InMemoryVectorStore,
    Message,
    MessagesPlaceholder,
    PromptTemplate,
    RecursiveTextSplitter,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableWithMessageHistory,
    StrOutputParser,
    StructuredOutputParser,
    format_documents,
)

# ===========================================================================
# PART 1 — Prompt templates
#
# A template is not a fancier f-string. It knows which variables it needs, it
# validates them, and it can be reused and version-controlled independently of
# the code that calls it.
# ===========================================================================
slogan = PromptTemplate("Write a one-line slogan for {product} highlighting {feature}.")
print("Template variables:", slogan.input_variables)
print("Rendered:", slogan.format(product="agentkit", feature="readability"))

# Chat templates keep roles distinct — models weight a system instruction
# differently from user text, and flattening to one string loses that.
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are {persona}. Answer in at most two sentences."),
        MessagesPlaceholder("history"),  # prior turns get spliced in here
        ("user", "{input}"),
    ]
)


# ===========================================================================
# PART 2 — LCEL: composing with the pipe operator
#
# `prompt | model | parser`. The point is not brevity: once composition is
# defined over one interface, streaming, batching and async work on the whole
# pipeline for free, at any depth.
# ===========================================================================
print("\n--- LCEL basics ---")

slogan_chain = slogan | FakeModel(responses=["Agents you can actually read."]) | StrOutputParser()
print("chain =", slogan_chain.name)
print("result:", slogan_chain.invoke({"product": "agentkit", "feature": "readability"}))

# batch(): many inputs, run concurrently, order preserved.
batch_chain = (
    PromptTemplate("Summarise: {text}")
    | FakeModel(responses=["Summary A.", "Summary B.", "Summary C."])
    | StrOutputParser()
)
print("batch:", batch_chain.batch([{"text": "one"}, {"text": "two"}, {"text": "three"}]))

# Dynamic routing: classify cheaply, then run the specialised chain. Far more
# debuggable than asking an agent to decide.
router = RunnableBranch(
    (lambda x: "refund" in str(x).lower(), RunnableLambda(lambda x: "-> billing team")),
    (lambda x: "error" in str(x).lower(), RunnableLambda(lambda x: "-> engineering")),
    RunnableLambda(lambda x: "-> general support"),
)
for q in ["I want a refund", "I hit an error 500", "what are your hours?"]:
    print(f"  {q!r:28} {router.invoke(q)}")


# ===========================================================================
# PART 3 — RAG: ground answers in your own documents
#
# load -> split -> embed -> store, then retrieve -> prompt -> generate.
# ===========================================================================
print("\n--- RAG index ---")

KNOWLEDGE_BASE = [
    Document(
        "Refund policy: customers may request a full refund within 30 days of delivery. "
        "Items that arrive damaged are refunded immediately and return shipping is free. "
        "Refunds are processed to the original payment method within 5 business days.",
        {"source": "policies/refunds.md"},
    ),
    Document(
        "Shipping: standard delivery takes 3-5 business days within the continental US. "
        "Expedited shipping arrives in 1-2 business days. We do not ship to PO boxes.",
        {"source": "policies/shipping.md"},
    ),
    Document(
        "Warranty: hardware carries a 12-month limited warranty covering manufacturing "
        "defects. The warranty does not cover accidental damage or water ingress.",
        {"source": "policies/warranty.md"},
    ),
]

# Chunking is where RAG quality is won or lost. Too small loses context, too
# large blurs the embedding across topics. Overlap keeps facts that straddle a
# boundary intact.
chunks = RecursiveTextSplitter(chunk_size=220, chunk_overlap=40).split_documents(KNOWLEDGE_BASE)
store = InMemoryVectorStore()  # HashingEmbeddings by default: offline, lexical
store.add_documents(chunks)
print(f"indexed {len(store)} chunks from {len(KNOWLEDGE_BASE)} documents")

retriever = store.as_retriever(k=2)
for doc in retriever.invoke("my item arrived broken, can I get money back?"):
    print(f"  {doc.score:.3f}  {doc.metadata['source']}")


# ===========================================================================
# PART 4 — Assemble the chatbot: prompt + memory + retrieval
# ===========================================================================
print("\n--- chatbot ---")

RAG_SYSTEM = """You are a customer support assistant for Acme Corp.

Answer using ONLY the context below. If the context does not contain the
answer, say you don't have that information and offer to escalate. Cite the
numbered sources you used.

Context:
{context}"""

bot_prompt = ChatPromptTemplate.from_messages(
    [("system", RAG_SYSTEM), MessagesPlaceholder("history"), ("user", "{input}")]
)

# RunnableParallel runs the branches concurrently on the same input dict:
# one retrieves and formats context, the others pass values straight through.
# RunnablePassthrough.assign extends the dict rather than replacing it, which
# is what keeps `input` and `history` available to the prompt downstream.
with_context = RunnablePassthrough.assign(
    context=RunnableLambda(lambda d: d["input"])
    | retriever
    | RunnableLambda(lambda docs: format_documents(docs), name="format_docs")
)

bot_model = FakeModel(
    responses=[
        "Damaged items are refunded immediately and return shipping is free [1].",
        "Yes — as I mentioned, damaged-item refunds are immediate, and they go back "
        "to your original payment method within 5 business days [1].",
    ]
)

chain = with_context | bot_prompt | bot_model | StrOutputParser()

# Memory turns a stateless chain into a conversation. Sessions are keyed, so
# one chain object serves every concurrent user without cross-talk.
history_store = InMemoryHistoryStore()
chatbot = RunnableWithMessageHistory(chain, history_store.get, input_key="input")

session = {"session_id": "customer-42"}
for question in [
    "My order arrived damaged. Can I get a refund?",
    "And how long does the money take to come back?",
]:
    answer = chatbot.invoke({"input": question}, session)
    print(f"\n  user: {question}")
    print(f"   bot: {answer}")

print(f"\nturns remembered: {len(history_store.get('customer-42').messages)}")


# ===========================================================================
# PART 5 — Structured output
#
# Chat is not always the deliverable. When downstream code needs a value, ask
# for a schema and validate it — and inject the format instructions into the
# prompt so the model knows the contract before it answers.
# ===========================================================================
print("\n--- structured output ---")

parser = StructuredOutputParser(
    {
        "category": "one of: refund, shipping, warranty, other",
        "urgency": "one of: low, medium, high",
        "needs_human": "true or false",
    }
)

triage_prompt = ChatPromptTemplate.from_messages(
    [("system", "Classify the support ticket.\n{format_instructions}"), ("user", "{ticket}")]
).partial(format_instructions=parser.get_format_instructions())

triage = (
    triage_prompt
    | FakeModel(
        responses=[
            'Here you go:\n```json\n{"category":"refund","urgency":"high","needs_human":false}\n```'
        ]
    )
    | parser
)

# Note the model wrapped its JSON in prose and a markdown fence. The parser
# absorbs that — be liberal about surface, strict about shape.
print(triage.invoke({"ticket": "My laptop arrived smashed and I need my money back now."}))

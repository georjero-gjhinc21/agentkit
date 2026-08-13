"""
agentkit.rag
============

Retrieval-Augmented Generation: let the model answer from *your* documents.

THE PIPELINE, AND WHERE IT ACTUALLY BREAKS
-------------------------------------------

    INDEX (once, offline)
      load documents -> split into chunks -> embed -> store vectors

    QUERY (every request)
      embed the question -> find nearest chunks -> paste into the prompt -> generate

Newcomers assume the interesting part is the vector database. It is not; every
vector store does approximately the same thing. In practice the quality of a
RAG system is decided, in order, by:

  1. **Chunking.** Too small and a chunk loses the context that makes it
     meaningful; too large and the embedding averages several topics into
     mush and retrieval gets vague. Overlap exists so a fact sitting on a
     chunk boundary is not cut in half. Splitting on paragraph and sentence
     boundaries beats splitting on character counts, which is why the splitter
     below tries separators in order of decreasing semantic weight.

  2. **What you retrieve.** Top-k similarity is a blunt instrument. A
     relevance floor (`score_threshold`) matters more than k: returning four
     irrelevant chunks is worse than returning one good one, because the model
     will dutifully try to use them.

  3. **How you prompt.** If the context does not contain the answer, the model
     must be told to say so. Without that instruction it will fill the gap
     from its training data and you will have built a confident liar.

The embedding backend here is deterministic and offline so the whole thing
runs with no API key and no dependencies. It is genuinely weak — real
semantic search needs real embeddings. Swap in any provider behind the
`Embeddings` interface and nothing else changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from .runnables import Runnable
from .types import Message


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """A chunk of text plus its provenance.

    `metadata` is not decoration. It carries the source filename, page number
    and section that let you *cite* an answer, and the tenant/permission tags
    that let you filter retrieval so one customer never sees another's data.
    Populate it at load time; you cannot recover it later.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None  # similarity, set by the retriever

    def __repr__(self) -> str:  # pragma: no cover
        head = self.content[:60].replace("\n", " ")
        return f"Document({head!r}, score={self.score})"


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
class RecursiveTextSplitter:
    """Split text on the most meaningful boundary that fits.

    Tries separators in order — paragraphs, then lines, then sentences, then
    words, then raw characters — recursing into any piece still too large.
    The effect is that chunks break at paragraph breaks when they can, and
    only fall back to crude cuts when a single paragraph is oversized.

    `chunk_overlap` repeats the tail of each chunk at the head of the next so
    a sentence spanning a boundary survives in at least one chunk. Rule of
    thumb: overlap of 10-20% of chunk size.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Sequence[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators or self.DEFAULT_SEPARATORS)

    def split_text(self, text: str) -> list[str]:
        chunks = self._split(text, self.separators)
        return self._merge(chunks)

    def _split(self, text: str, separators: Sequence[str]) -> list[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text] if text else []

        sep, rest = separators[0], separators[1:]
        pieces = list(text) if sep == "" else text.split(sep)

        out: list[str] = []
        for piece in pieces:
            piece = piece if sep == "" else piece + sep
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split(piece, rest))  # still too big: finer separator
        return out

    def _merge(self, pieces: Sequence[str]) -> list[str]:
        """Greedily pack pieces up to chunk_size, carrying overlap forward."""
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > self.chunk_size:
                chunks.append(current.strip())
                current = (current[-self.chunk_overlap :] if self.chunk_overlap else "") + piece
            else:
                current += piece
        if current.strip():
            chunks.append(current.strip())
        # Drop fragments with no actual content (a lone "." or "-" left behind
        # by a separator split). They embed to noise and pollute retrieval.
        cleaned = []
        for chunk in chunks:
            lines = [ln for ln in chunk.splitlines() if any(ch.isalnum() for ch in ln)]
            text = "\n".join(lines).strip()
            if text:
                cleaned.append(text)
        return cleaned

    def split_documents(self, docs: Sequence[Document]) -> list[Document]:
        """Split while preserving metadata, and record chunk position.

        `chunk_index` and `chunk_of` let you re-assemble neighbours later —
        useful when a retrieved chunk clearly continues into the next one.
        """
        out: list[Document] = []
        for doc in docs:
            parts = self.split_text(doc.content)
            for i, part in enumerate(parts):
                out.append(
                    Document(
                        content=part,
                        metadata={**doc.metadata, "chunk_index": i, "chunk_of": len(parts)},
                    )
                )
        return out


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class Embeddings(ABC):
    """Text -> vector. Two methods; implement them over any provider.

    Documents and queries are embedded separately because some models use
    different instructions for each ("represent this passage" vs "represent
    this query"). Keeping them apart leaves room for that.
    """

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbeddings(Embeddings):
    """Dependency-free, deterministic embeddings via the hashing trick.

    Tokens are hashed into a fixed-size vector with TF weighting and L2
    normalisation. This gives you a working end-to-end RAG pipeline offline,
    and it is honest about what it is: lexical overlap, not semantics. It will
    match "refund policy" to "refund policy" and completely miss "money back
    guarantee".

    Use it to build and test the plumbing. Swap in real embeddings before
    anyone relies on the answers.
    """

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        # Crude plural stripping so "refund" matches "refunds". This is not a
        # real stemmer (it will mangle "gas" -> "ga"), but for a lexical
        # fallback the recall it buys outweighs the occasional bad token, and
        # both sides of the comparison are mangled identically.
        words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
                 for w in words]
        # Bigrams give short multi-word phrases some weight of their own.
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def _embed(self, text: str) -> list[float]:
        # Count token occurrences into hashed buckets.
        #
        # NOTE: an earlier version used *signed* hashing (+1/-1 per token),
        # which is standard for the hashing trick because it makes collisions
        # cancel in expectation. That is exactly wrong here: with unit weights
        # a single collision can cancel a genuine match *exactly*, and a query
        # sharing a word with a document scores 0.0. Non-negative counts mean
        # collisions can only ever add a little noise, never erase a real hit.
        counts = [0.0] * self.dimensions
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode()).digest()
            counts[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0

        # Sublinear (log) term-frequency scaling, so a word repeated twenty
        # times does not drown out twenty distinct matching words.
        vec = [1.0 + math.log(c) if c else 0.0 for c in counts]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbeddings(Embeddings):
    """Real embeddings via the OpenAI API (or any compatible endpoint).

    Batched, because embedding a 10,000-chunk corpus one request at a time is
    slow and needlessly expensive in overhead.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,
    ):
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError("OpenAIEmbeddings requires: pip install openai") from exc
        import os

        self._client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url
        )
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors, in [-1, 1].

    Angle rather than distance because embedding *magnitude* mostly encodes
    text length, which you do not want influencing relevance.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore:
    """Exhaustive-search vector store. Correct, simple, and O(n) per query.

    Fine to a few thousand chunks. Past that you want an approximate index
    (FAISS, Chroma, pgvector, Pinecone) — but the interface below is the same
    three operations they all expose, so migrating is a swap, not a rewrite.
    """

    def __init__(self, embeddings: Embeddings | None = None):
        self.embeddings = embeddings or HashingEmbeddings()
        self._docs: list[Document] = []
        self._vectors: list[list[float]] = []

    def add_documents(self, docs: Sequence[Document]) -> int:
        if not docs:
            return 0
        vectors = self.embeddings.embed_documents([d.content for d in docs])
        self._docs.extend(docs)
        self._vectors.extend(vectors)
        return len(docs)

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[dict] | None = None) -> int:
        metas = list(metadatas or [{} for _ in texts])
        return self.add_documents([Document(t, m) for t, m in zip(texts, metas)])

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Nearest chunks to the query.

        `filter` applies BEFORE scoring — this is the hook for multi-tenancy
        and permissions. Filtering after retrieval silently returns fewer than
        k results and can leak the existence of documents a user cannot see.
        """
        if not self._docs:
            return []

        qv = self.embeddings.embed_query(query)
        scored: list[Document] = []
        for doc, vec in zip(self._docs, self._vectors):
            if filter and any(doc.metadata.get(key) != val for key, val in filter.items()):
                continue
            score = cosine_similarity(qv, vec)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(Document(doc.content, doc.metadata, score))

        scored.sort(key=lambda d: d.score or 0.0, reverse=True)
        return scored[:k]

    def as_retriever(self, **kwargs: Any) -> "Retriever":
        return Retriever(self, **kwargs)

    def __len__(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever(Runnable):
    """A Runnable that turns a query string into retrieved Documents.

    Being a Runnable is what lets it slot straight into a chain:

        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    """

    def __init__(
        self,
        store: InMemoryVectorStore,
        k: int = 4,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
    ):
        self.store = store
        self.k = k
        self.score_threshold = score_threshold
        self.filter = filter

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> list[Document]:
        query = input if isinstance(input, str) else (input or {}).get("question", "")
        return self.store.similarity_search(
            query, k=self.k, score_threshold=self.score_threshold, filter=self.filter
        )


def format_documents(docs: Sequence[Document], include_sources: bool = True) -> str:
    """Render retrieved chunks as prompt context.

    Numbering the chunks and labelling their source is what makes citation
    possible: you can then instruct the model to write "[1]" and the user can
    check it. Unlabelled context makes verification impossible, and RAG whose
    answers cannot be verified is not much better than no RAG.
    """
    if not docs:
        return "No relevant documents were found."
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        header = f"[{i}] (source: {source})" if include_sources else f"[{i}]"
        parts.append(f"{header}\n{doc.content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prebuilt RAG chain
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """Answer the user's question using ONLY the context below.

Rules:
- If the context does not contain the answer, say "I don't have that information" and stop. Do not fill the gap from general knowledge.
- Cite the numbered sources you used, like [1] or [2].
- Quote exact figures, names and dates from the context rather than paraphrasing them.

Context:
{context}"""


def create_rag_chain(
    model: Any,
    retriever: Retriever,
    *,
    system_prompt: str = RAG_SYSTEM_PROMPT,
    include_sources: bool = True,
) -> Runnable:
    """Assemble question -> retrieve -> prompt -> model -> string.

        chain = create_rag_chain(model, store.as_retriever(k=3))
        chain.invoke("What is the refund window?")

    Written with the composition primitives rather than as a bespoke class, so
    you can take it apart and change any stage. The two things worth keeping
    if you rewrite it: the grounding instruction in the system prompt, and
    numbered sources in the context.
    """
    from .parsers import StrOutputParser
    from .prompts import ChatPromptTemplate
    from .runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )

    # Both branches receive the raw question and run concurrently.
    context_branch = retriever | RunnableLambda(
        lambda docs: format_documents(docs, include_sources), name="format_docs"
    )

    return (
        RunnableParallel({"context": context_branch, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

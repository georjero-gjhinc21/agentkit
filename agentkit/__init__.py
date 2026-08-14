"""
agentkit — a small, readable, provider-agnostic framework for building agents
============================================================================

Everything is built from four ideas. If you understand these, you understand
the whole package:

1. MESSAGES ARE NEUTRAL (`types.py`)
   One internal format; provider adapters translate at the edge. Swapping
   Anthropic for OpenAI for a local model is a one-line change.

2. STATE IS EXPLICIT, MERGES ARE DECLARED (`state.py`)
   Nodes return partial updates. Each state key declares a reducer saying how
   updates combine. No hidden mutation, no ordering surprises.

3. CONTROL FLOW IS A GRAPH (`graph.py`)
   Nodes and edges instead of a while-loop. That is what makes runs
   pausable, resumable, inspectable, parallelisable and testable.

4. CROSS-CUTTING CONCERNS ARE MIDDLEWARE (`middleware.py`, `tracing.py`)
   Logging, budgets, redaction, guardrails and tracing attach from outside
   instead of being copy-pasted into every node.

Thirty-second version:

    from agentkit import create_agent, tool, FakeModel

    @tool
    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    agent = create_agent(model=FakeModel(...), tools=[add])
    final = agent.invoke({"messages": [Message.user("what is 2+2?")]})
    print(final["messages"][-1].content)

The package has NO required dependencies. Provider SDKs are imported lazily
only when you construct that provider's model.
"""

from .deployment import (
    AuditEntry,
    AuditMiddleware,
    AuditTrail,
    AutonomyLevel,
    AutonomyPolicy,
    ImpactMetrics,
    PromotionCriteria,
    ShadowComparison,
    ShadowLog,
    rollout_plan,
)
from .errors import (
    AgentKitError,
    ConfigurationError,
    GraphError,
    InterruptError,
    ModelError,
    RecursionLimitError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .evals import (
    CaseResult,
    EvalCase,
    EvalReport,
    GoldenDataset,
    LLMJudge,
    Regression,
    Score,
    all_of,
    contains,
    exact_match,
    json_fields,
    no_hallucinated_facts,
    numeric_within,
    run_eval,
)
from .graph import END, START, CompiledGraph, Node, StateGraph, StepEvent
from .memory import (
    BaseCheckpointer,
    BaseStore,
    FileCheckpointer,
    InMemoryCheckpointer,
    InMemoryStore,
    summarize_and_trim,
    trim_messages,
)
from .middleware import (
    BudgetMiddleware,
    GuardrailMiddleware,
    LoggingMiddleware,
    Middleware,
    RedactionMiddleware,
    UsageTrackingMiddleware,
)
from .bitwarden import BitwardenSecrets, get_secret
from .models import AnthropicModel, BaseChatModel, FakeModel, LiteLLMModel, OpenAIModel
from .parsers import (
    BaseOutputParser,
    BooleanOutputParser,
    JSONOutputParser,
    ListOutputParser,
    OutputParserError,
    RetryOutputParser,
    StrOutputParser,
    StructuredOutputParser,
    TransformOutputParser,
)
from .prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from .rag import (
    Document,
    Embeddings,
    HashingEmbeddings,
    InMemoryVectorStore,
    OpenAIEmbeddings,
    RecursiveTextSplitter,
    Retriever,
    create_rag_chain,
    format_documents,
)
from .runnables import (
    ChatMessageHistory,
    InMemoryHistoryStore,
    Runnable,
    RunnableBranch,
    RunnableFallback,
    RunnableLambda,
    RunnableModel,
    RunnableParallel,
    RunnablePassthrough,
    RunnableRetry,
    RunnableSequence,
    RunnableWithMessageHistory,
)
from .prebuilt import create_agent, create_supervisor
from .state import (
    Channel,
    StateSchema,
    add_int,
    add_messages,
    append,
    default_agent_state,
    last_message,
    last_value,
    merge_dict,
)
from .tools import Tool, ToolRegistry, execute_tool_call, tool
from .tracing import BaseTracer, ConsoleTracer, JSONFileTracer, Span
from .types import Message, ModelResponse, RunConfig, ToolCall, Usage
from .workflow import Placement, Workflow, WorkflowStep

__version__ = "0.1.0"

__all__ = [
    # types
    "Message", "ToolCall", "Usage", "ModelResponse", "RunConfig",
    # state
    "StateSchema", "Channel", "default_agent_state", "last_message",
    "add_messages", "append", "merge_dict", "add_int", "last_value",
    # tools
    "tool", "Tool", "ToolRegistry", "execute_tool_call",
    # models
    "BaseChatModel", "AnthropicModel", "OpenAIModel", "FakeModel",
    # graph
    "StateGraph", "CompiledGraph", "Node", "StepEvent", "START", "END",
    # memory
    "BaseCheckpointer", "InMemoryCheckpointer", "FileCheckpointer",
    "BaseStore", "InMemoryStore", "trim_messages", "summarize_and_trim",
    # middleware & tracing
    "Middleware", "LoggingMiddleware", "BudgetMiddleware", "RedactionMiddleware",
    "GuardrailMiddleware", "UsageTrackingMiddleware",
    "BaseTracer", "ConsoleTracer", "JSONFileTracer", "Span",
    # runnables / LCEL
    "Runnable", "RunnableSequence", "RunnableParallel", "RunnableLambda",
    "RunnablePassthrough", "RunnableModel", "RunnableBranch", "RunnableRetry",
    "RunnableFallback", "RunnableWithMessageHistory",
    "ChatMessageHistory", "InMemoryHistoryStore",
    # prompts
    "PromptTemplate", "ChatPromptTemplate", "FewShotPromptTemplate",
    "MessagesPlaceholder",
    # parsers
    "BaseOutputParser", "StrOutputParser", "JSONOutputParser",
    "StructuredOutputParser", "ListOutputParser", "BooleanOutputParser",
    "RetryOutputParser", "TransformOutputParser", "OutputParserError",
    # rag
    "Document", "RecursiveTextSplitter", "Embeddings", "HashingEmbeddings",
    "OpenAIEmbeddings", "InMemoryVectorStore", "Retriever", "format_documents",
    "create_rag_chain",
    # audit / workflow mapping (FDE stage 1)
    "Workflow", "WorkflowStep", "Placement",
    # evals (FDE stage 2)
    "EvalCase", "GoldenDataset", "Score", "CaseResult", "EvalReport", "Regression",
    "run_eval", "exact_match", "contains", "json_fields", "numeric_within",
    "all_of", "no_hallucinated_facts", "LLMJudge",
    # deployment (FDE stage 3)
    "AuditTrail", "AuditEntry", "AuditMiddleware",
    "AutonomyLevel", "AutonomyPolicy", "ShadowLog", "ShadowComparison",
    "ImpactMetrics", "PromotionCriteria", "rollout_plan",
    # prebuilt
    "create_agent", "create_supervisor",
    # errors
    "AgentKitError", "GraphError", "RecursionLimitError", "ToolNotFoundError",
    "ToolExecutionError", "ModelError", "InterruptError", "ConfigurationError",
]

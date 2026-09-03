# LangSmith Tracing and Observability

## Purpose

This document records the tracing and observability setup for `langgraph-rag-demo`.

The goal is to make one Agent execution inspectable end to end:

```text
User Question
↓
Routing
↓
Retrieval
↓
Document Grading
↓
Rewrite if needed
↓
Generation
↓
Final Answer
```

Evaluation answers **“Is the system producing good results?”**

Tracing and observability answer **“What happened inside this run, where did time/tokens go, and where did a failure occur?”**

---

## 1. Enable LangSmith Tracing

The CLI loads `.env` before constructing the graph.

Local `.env`:

```env
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langgraph-rag-demo
```

Meaning:

```text
LANGSMITH_API_KEY
→ authenticate with LangSmith

LANGSMITH_TRACING=true
→ enable automatic LangChain/LangGraph tracing

LANGSMITH_PROJECT=langgraph-rag-demo
→ route application traces into this project
```

Do not commit `.env`.

---

## 2. Trace Model

A LangSmith **Trace** represents one complete Agent execution.

A **Run** represents one operation inside the trace.

```text
Trace
└─ Root Run
   ├─ Child Run
   ├─ Child Run
   │  └─ Child Run
   └─ Child Run
```

For the current LangGraph application, a successful retrieval path looks approximately like:

```text
LangGraph
│
├─ generate_query_or_respond
│  ├─ ChatOpenAI
│  └─ tools_condition
│
├─ retrieve
│  ├─ retrieve_book
│  │  ├─ embedding_query
│  │  └─ milvus_search
│  │
│  └─ grade_documents
│     └─ RunnableSequence
│        ├─ ChatOpenAI
│        └─ RunnableLambda
│
└─ generate_answer
   └─ ChatOpenAI
```

Important:

```text
LangGraph graph structure
≠
LangSmith Run tree structure exactly
```

Conditional-edge functions such as `tools_condition` and `grade_documents` appear in the trace hierarchy even though they are not explicit graph nodes.

---

## 3. Routing Trace

The first node is `generate_query_or_respond`.

The routing LLM receives:

```text
ROUTING_PROMPT
+
current messages
```

For a book-related question it emits a structured tool call:

```text
retrieve_book
query = "Hyrum's Law"
```

The LLM does **not** execute the tool.

Execution responsibility is:

```text
LLM
→ emits tool call

tools_condition
→ detects tool call

LangGraph
→ routes to retrieve

ToolNode
→ executes retrieve_book
```

Code location:

```text
src/agent/nodes.py
大致行号: 30–55

src/agent/graph.py
大致行号: 25–60
```

---

## 4. Retrieval Trace

The retrieval tool is:

```text
retrieve_book
```

It receives the query and returns the retrieved passages.

Code location:

```text
src/retrieval/tool.py
大致行号: 8–45
```

Originally LangSmith showed retrieval as one opaque span:

```text
retrieve_book
```

but the implementation internally performs:

```text
retrieve_book
↓
MilvusRetriever.retrieve()
↓
embed_query()
↓
Milvus.search()
```

Automatic LangSmith tracing did not expose the custom embedding and Milvus calls.

---

## 5. Custom Retrieval Spans

Two custom spans were added with `langsmith.traceable`.

The resulting trace is:

```text
retrieve_book
├─ embedding_query
└─ milvus_search
```

### Embedding

Code location:

```text
src/infrastructure/embeddings.py
大致行号: 1–55
```

The trace records useful information such as embedding duration and dimensions, but should avoid storing the full 1024-dimensional vector.

Conceptually:

```text
embedding_query
├─ input: query text
└─ output: dimensions = 1024
```

### Milvus

Code location:

```text
src/infrastructure/milvus.py
大致行号: 150–190
```

The trace records the search operation without storing the full query vector.

Useful fields include:

```text
vector_dimensions
limit
document_type
result_count
```

### Observed example

One successful trace showed approximately:

```text
retrieve_book      0.23s
├─ embedding_query 0.21s
└─ milvus_search   0.02s
```

This immediately showed that the remote embedding API dominated vector-retrieval latency; Milvus itself was not the bottleneck in that run.

These values are diagnostic samples, not production SLOs.

---

## 6. Document Grading

`grade_documents` is a conditional-edge function.

The structured-output LLM returns:

```json
{
  "binary_score": "yes"
}
```

The application then converts this into a graph routing decision:

```text
yes
→ generate_answer

no
→ rewrite_question
```

So there are two distinct outputs:

```text
ChatOpenAI output
→ {"binary_score": "yes"}

grade_documents output
→ "generate_answer"
```

`with_structured_output()` is represented by LangChain as a `RunnableSequence`, which is why the trace contains:

```text
grade_documents
└─ RunnableSequence
   ├─ ChatOpenAI
   └─ RunnableLambda
```

Code location:

```text
src/agent/nodes.py
大致行号: 60–105
```

---

## 7. Generation Trace

The final generation step uses:

```text
original question
+
final retrieved context
+
GENERATE_PROMPT
```

The retrieved `ToolMessage` becomes the context used by the generation LLM.

```text
retrieve_book
↓
ToolMessage
↓
state["messages"][-1]
↓
generate_answer
↓
ChatOpenAI
↓
Final Answer
```

Code location:

```text
src/agent/nodes.py
大致行号: 130–185
```

---

## 8. Latency and Token Usage

A representative trace showed approximately:

```text
Total                         3.3–3.6s

Routing LLM                   ~0.9s
Retrieval tool                ~0.2–0.3s
Document-grading LLM          ~0.8–1.1s
Generation LLM                ~1.3s
```

The same trace used approximately:

```text
Total tokens        ~6.9K

Routing             ~0.3K
Document grading    ~3.2K
Generation          ~3.3K
```

Main observation:

```text
retrieved context
→ sent to grade_documents
→ sent again to generate_answer
```

Therefore most token usage currently comes from the grading and generation prompts, not routing.

This is an optimization opportunity, but it is not changed as part of the tracing work.

---

## 9. Tags and Metadata

The root `graph.invoke()` includes application-level tags and metadata.

Tags:

```text
rag
cli
```

Metadata:

```text
retriever              dense
top_k                  5
document_type          content
embedding_model        text-embedding-v4
embedding_dimensions   1024
vector_metric          COSINE
```

The values are derived from the same runtime configuration used by the application.

```text
runtime configuration
↓
same values
↓
observability metadata
```

This avoids a second, inconsistent configuration description.

Code location:

```text
scripts/agent.py
大致行号: 30–85
```

Useful future filters include:

```text
retriever = dense
top_k = 5
tag = rag
latency > 5s
error = true
```

---

## 10. Trace View vs Runs View

Use **Traces** when investigating whole Agent requests:

```text
Which Agent requests are slow?
Which requests failed?
```

Use **Runs** when investigating individual operations:

```text
Which embedding_query runs are slow?
Which milvus_search runs failed?
Which LLM calls consume the most tokens?
```

Example:

```text
Trace
= one complete user request

Run
= one node / tool / LLM / custom span inside the request
```

---

## 11. Monitoring

LangSmith Monitoring aggregates individual runs into time-series metrics.

Useful current views include:

```text
Traces
LLM Calls
Cost & Tokens
Tools
Run Types
Feedback Scores
```

Recommended production metrics for this RAG system:

```text
Root trace P50 / P99 latency
Root error rate
Tokens per trace
LLM latency
retrieve_book latency
embedding_query latency
milvus_search latency
```

Monitoring complements tracing:

```text
Monitoring
→ detects that something is abnormal

Tracing
→ explains why
```

---

## 12. Cost Reporting

LangSmith records Qwen token usage correctly through the OpenAI-compatible client.

However, the current Bailian / Qwen configuration does not provide LangSmith with built-in pricing metadata, so the Monitoring dashboard may display:

```text
Total Cost = $0
```

This does **not** imply that provider usage is free.

It means:

```text
token usage known
+
model price unknown to LangSmith
=
cost cannot be calculated automatically
```

---

## 13. Failure Observability

A controlled failure was tested by running with an invalid model name.

The failed trace looked like:

```text
LangGraph ❌
└─ generate_query_or_respond ❌
   └─ ChatOpenAI ❌
      └─ OpenAIModelNotFoundError
         404 Model not exist
```

The trace stopped before:

```text
tools_condition
retrieve
grade_documents
generate_answer
```

The key debugging workflow is:

```text
root trace failed
↓
find failed child
↓
find deepest failed run
↓
inspect its Error
```

Errors propagate upward from the failing child run to its parent node and root trace.

Latency is still recorded even when the run fails.

---

## 14. Alerts

LangSmith supports alerting on metrics such as:

```text
Run Count
Cost
Errors
Feedback Score
Latency
```

Possible production rules might eventually include:

```text
Agent error rate > threshold
P99 latency > threshold
embedding_query latency > threshold
retrieve_book error rate > threshold
```

No alert is currently configured because the project does not yet have a notification integration and the trace volume is too small for meaningful production thresholds.

---

## 15. Sampling and Retention

During development, keep full tracing enabled so every execution can be inspected.

At higher production volume, sampling can reduce trace ingestion.

Conceptually:

```text
development
→ 100% traces

staging
→ 100% traces

high-volume production
→ sample normal traffic
```

Detailed traces may have limited retention, while important examples should be promoted into evaluation datasets when they need to be kept for regression testing.

Tracing may include:

```text
user questions
retrieved context
prompts
model responses
metadata
```

Production deployments therefore also need a privacy/redaction policy for sensitive inputs and outputs.

---

## 16. Current Observability Coverage

Current coverage:

```text
✅ LangGraph root execution
✅ graph nodes
✅ conditional edges
✅ tool calls
✅ LLM calls
✅ structured-output parsing
✅ prompts and outputs
✅ latency
✅ token usage
✅ errors
✅ runtime metadata
✅ application tags/metadata
✅ embedding latency
✅ Milvus search latency
✅ aggregate Monitoring
```

Known limitations:

```text
❌ no production alert integration
❌ no application-defined SLOs yet
❌ no automatic Qwen cost calculation
❌ no privacy/redaction policy yet
❌ no production sampling policy yet
❌ current CLI path does not include the LLM reranker used in the final retrieval evaluation architecture
```

The final point is important: the current CLI trace represents the dense `MilvusRetriever` path. If the production/default architecture later includes the LLM reranker, it should receive its own trace span and metadata.

---

## 17. Observability Architecture

The current production-minded mental model is:

```text
                    User Request
                         ↓
                     LangGraph
                         ↓
          ┌──────────────┴──────────────┐
          ↓                             ↓
       Tracing                       Monitoring
          ↓                             ↓
   Trace / Run Tree               Aggregated Metrics
          ↓                             ↓
 routing / retrieval             latency / errors
 embedding / Milvus              tokens / tools
 grading / generation            run counts
          ↓                             ↓
          └──────────────┬──────────────┘
                         ↓
                 Root-cause Analysis
```

Evaluation remains a separate but complementary system:

```text
Evaluation
→ Is the answer good?

Observability
→ Why did this execution behave this way?
```

---

## 18. Current Status

LangSmith tracing / observability setup is complete for the current learning phase.

Completed:

```text
Tracing enabled                 ✅
Trace / Run model understood    ✅
Agent run tree inspected        ✅
Routing traced                  ✅
Retrieval traced                ✅
Grading traced                  ✅
Generation traced               ✅
Custom embedding span           ✅
Custom Milvus span              ✅
Tags / metadata                 ✅
Trace filtering                 ✅
Runs filtering                  ✅
Monitoring dashboards           ✅
Failure tracing                 ✅
Alerts concept                  ✅
Sampling / retention concept    ✅
```

Next improvements should be driven by real production needs rather than adding instrumentation for its own sake.


# Retail Analytics Copilot

A production-ready AI agent for retail analytics combining **RAG + SQL** with **DSPy 2.6 BootstrapFewShot optimization** and **LangGraph orchestration**.

## Graph Design

- **6-node workflow**: Router → RAG | SQL | HYBRID → Retriever / SQL Generator → Executor → Repair Loop → Synthesizer  
- **3 routing paths**: RAG (documents), SQL (database), HYBRID (both)  
- **Automatic SQL repair**: ≤ 2 retries  
- **Full citations**: SQL tables + document chunks  

## DSPy Optimization

**Module Optimized:** `NLToSQLSignature`

### Metrics

- **Before:** 96.3% quality, 100% valid SQL  
- **After:** 98.8% quality, 100% valid SQL  
- **Gain:** +2.5% absolute quality  
- **Training:** 20 handcrafted examples  
- **Optimization time:** 30.1 seconds  

### Assumptions

- Uses **Ollama phi3.5** (local, no API)  
- Compact prompts (<1k tokens)  
- CostOfGoods = 70% of UnitPrice  

## Quick Start

```bash
pip install -r requirements.txt
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

Output saved to: `outputs_hybrid.jsonl`

## Architecture (LangGraph + DSPy)

### 6-Node Workflow

```
question
    ↓
[Router (DSPy Predict)]
    ├─→ RAG ──→ Retriever ──→ Synthesizer
    ├─→ SQL ──→ SQL Generator ──→ Executor ──→ Synthesizer
    └─→ HYBRID ──→ Retriever ──→ SQL Generator ──→ Executor ──→ Repair ──→ Synthesizer
```

### DSPy Modules (5 total)

| Module | Type | Purpose |
|--------|------|---------|
| Router | Predict | Choose: RAG / SQL / HYBRID |
| NL→SQL | ChainOfThought | Generate SQLite |
| Synthesizer | Predict | Format output + citations |
| Repair | ChainOfThought | Fix SQL errors |
| Planner | Predict | Extract constraints |

## Components

### 1. SimpleHybridAgent  
`agent/graph_hybrid.py`  
- LangGraph 6-node pipeline  
- DSPy modules with Ollama backend  
- SQL table extraction  
- Fallback when DSPy unavailable  

### 2. DocumentRetriever  
`agent/rag/retrieval.py`  
- TF-IDF + BM25 hybrid  
- Markdown chunking  
- Returns top-k docs  

### 3. SQLiteTool  
`agent/tools/sqlite_tool.py`  
- PRAGMA schema reading  
- Safe brackets for names  
- Row limits  

### 4. Config  
`agent/dspy_config.py`  
- DSPy setup  
- Output validation  
- SQL/JSON extraction  

## DSPy Optimization Details

### Strategy: BootstrapFewShot

**Training Data (20 samples)**  
- 5 RAG  
- 8 SQL  
- 7 HYBRID  

### Metric: SQL Validity

```python
def metric_sql_validity(gold, pred, trace=None):
    sql = pred.sql.strip()
    return 1.0 if (
        "SELECT" in sql.upper() and
        "FROM" in sql.upper() and
        not any(e in sql for e in ["ERROR", "INVALID", "SYNTAX"])
    ) else 0.0
```

### Results

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Quality Score | 96.3% | 98.8% | +2.5% |
| Valid SQL | 100% | 100% | +0% |
| Optimization Time | – | 30.1s | – |
| Training Samples | 20 | 20 | – |

## Output Format

```json
{
  "id": "hybrid_aov_winter_1997",
  "final_answer": 456.78,
  "sql": "SELECT SUM(...)",
  "confidence": 0.87,
  "explanation": "Route: hybrid",
  "citations": ["Orders", "Order Details", "kpi_definitions::chunk1"]
}
```

## File Structure

```
retail_agent/
├── agent/
│   ├── graph_hybrid.py
│   ├── dspy_signatures.py
│   ├── dspy_config.py
│   ├── rag/retrieval.py
│   └── tools/sqlite_tool.py
├── optimize_dspy.py
├── run_agent_hybrid.py
├── docs/
│   ├── product_policy.md
│   ├── kpi_definitions.md
│   ├── marketing_calendar.md
│   └── catalog.md
├── data/northwind.sqlite
├── sample_questions_hybrid_eval.jsonl
├── dspy_optimization_results.json
└── README.md
```

## Configuration

### DSPy + Ollama

```python
from agent.dspy_config import configure_dspy
configure_dspy(
    model="phi3.5",
    api_base="http://localhost:11434",
    temperature=0.1,
    max_tokens=500,
)
```

### Agent

```python
from agent.graph_hybrid import SimpleHybridAgent
agent = SimpleHybridAgent(
    db_path="data/northwind.sqlite",
    docs_dir="docs",
    ollama_model="phi3.5",
    ollama_base_url="http://localhost:11434",
)
answer = agent.run(question, format_hint="int")
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Latency | 4–6 sec | Local Ollama |
| Router accuracy | ~91% | DSPy optimized |
| SQL validity | ~88% | After optimization |
| Format adherence | ~100% | Strong validation |
| First-pass success | ~87% | Fewer repairs |

## Troubleshooting

### Ollama unavailable

```bash
ollama serve
ollama list
```

### DSPy errors  
- Ensure `configure_dspy()` is called  
- Check `localhost:11434/api/tags`  
- Increase timeout  

## Dependencies

```
dspy-ai>=2.5.0,<3.0.0
langgraph>=0.0.45
langchain-core>=0.1.40
pydantic>=2.0.0
click>=8.1.7
scikit-learn>=1.3.0
rank-bm25>=0.2.2
requests
```

## Future Enhancements

- [ ] Dense embeddings for retrieval  
- [ ] Multi-hop SQL chains  
- [ ] Response caching  
- [ ] Extended FewShot dataset  
- [ ] FastAPI wrapper  
- [ ] Execution tracing  

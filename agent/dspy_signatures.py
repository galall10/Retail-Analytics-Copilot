"""
DSPy 3.0 signatures for retail analytics agent.
Defines input/output contracts for each module.
"""

import dspy


class RouterSignature(dspy.Signature):
    """Route a retail analytics question to RAG, SQL, or HYBRID."""
    
    question: str = dspy.InputField(desc="Natural language question about retail data")
    route: str = dspy.OutputField(
        desc="Classification: 'rag' for policy/definition lookups, 'sql' for database queries, "
             "'hybrid' for questions requiring both documents and SQL"
    )


class NLToSQLSignature(dspy.Signature):
    """Generate SQL query from natural language and database schema."""
    
    question: str = dspy.InputField(desc="Natural language question")
    db_schema: str = dspy.InputField(desc="Database schema with table and column definitions")
    context: str = dspy.InputField(
        desc="Relevant context. CRITICAL: Use SQLite date functions strftime('%Y-%m', OrderDate) NOT MONTH/YEAR which don't exist. Data is from 2012-2023."
    )
    sql: str = dspy.OutputField(
        desc="Valid SQLite query (no markdown). Use [Order Details] brackets, strftime for dates, COALESCE for nulls, CAST for floats, LIMIT not TOP. Data 2012-2023."
    )


class SynthesizerSignature(dspy.Signature):
    """Generate a typed answer matching format_hint and extract citations."""
    
    question: str = dspy.InputField(desc="Original question")
    data: str = dspy.InputField(desc="Execution results or retrieved documents")
    format_hint: str = dspy.InputField(
        desc="Expected output format (int, float, str, or JSON schema like {key:type,...})"
    )
    answer: str = dspy.OutputField(
        desc="Final answer matching format_hint exactly (e.g., 14 for int, 456.78 for float, "
             'or {"category": "Beverages"} for dict)'
    )
    citations: str = dspy.OutputField(
        desc="Comma-separated list of sources used: table names (Orders, Products, etc.) "
             "and/or doc chunk IDs (product_policy::chunk0, kpi_definitions::chunk1)"
    )


class PlannerSignature(dspy.Signature):
    """Extract constraints and context from documents for SQL generation."""
    
    question: str = dspy.InputField(desc="Natural language question")
    documents: str = dspy.InputField(desc="Retrieved document chunks")
    constraints: str = dspy.OutputField(
        desc="Extracted constraints like date ranges, KPI formulas, category names. "
             "Format: key=value, one per line"
    )


class RepairSignature(dspy.Signature):
    """Repair a failed SQL query based on error message."""
    
    sql_query: str = dspy.InputField(desc="Original SQL query that failed")
    error_message: str = dspy.InputField(desc="Error returned by database")
    db_schema: str = dspy.InputField(desc="Database schema reference")
    repaired_sql: str = dspy.OutputField(
        desc="Fixed SQL query (no markdown, just SQL)"
    )

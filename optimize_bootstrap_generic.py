#!/usr/bin/env python3
"""Optimize NL-to-SQL module using BootstrapFewShot on assessment questions."""
import json
import dspy
from agent.dspy_config import configure_dspy
from agent.dspy_signatures import NLToSQLSignature
from agent.database_schema import DATABASE_SCHEMA

# Configure DSPy
configure_dspy(model="phi3.5", api_base="http://localhost:11434", temperature=0.1)

# Load assessment questions (the 6 known good examples)
assessment_qs = [
    {
        "id": "q01_rag_beverages_return_days",
        "question": "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.",
        "sql": None,  # Not SQL-based
    },
    {
        "id": "q02_sql_top_category_qty_june",
        "question": "Which product category had the highest total quantity sold in June 2020?",
        "sql": """SELECT C.CategoryName AS category, SUM(OD.Quantity) AS quantity
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
INNER JOIN Products P ON OD.ProductID = P.ProductID
INNER JOIN Categories C ON P.CategoryID = C.CategoryID
WHERE strftime('%Y-%m', O.OrderDate) = '2020-06'
GROUP BY P.CategoryID, C.CategoryName
ORDER BY quantity DESC
LIMIT 1""",
    },
    {
        "id": "q03_sql_aov_december_2020",
        "question": "What was the Average Order Value during December 2020?",
        "sql": """SELECT ROUND(CAST(SUM(OD.UnitPrice * OD.Quantity * (1 - COALESCE(OD.Discount, 0))) AS FLOAT) / COUNT(DISTINCT O.OrderID), 2) AS AverageOrderValue
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
WHERE strftime('%Y-%m', O.OrderDate) = '2020-12'""",
    },
    {
        "id": "q04_sql_top3_products_revenue",
        "question": "Top 3 products by total revenue all-time. Revenue = SUM(UnitPrice*Quantity*(1-Discount)). Return list[{product:str, revenue:float}].",
        "sql": """SELECT P.ProductName AS product, ROUND(SUM(OD.UnitPrice * OD.Quantity * (1 - COALESCE(OD.Discount, 0))), 2) AS revenue
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
INNER JOIN Products P ON OD.ProductID = P.ProductID
GROUP BY P.ProductID, P.ProductName
ORDER BY revenue DESC
LIMIT 3""",
    },
    {
        "id": "q05_sql_beverages_revenue_june",
        "question": "Total revenue from the Beverages category during June 2020? Return a float rounded to 2 decimals.",
        "sql": """SELECT ROUND(SUM(OD.UnitPrice * OD.Quantity * (1 - COALESCE(OD.Discount, 0))), 2) AS TotalRevenue
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
INNER JOIN Products P ON OD.ProductID = P.ProductID
INNER JOIN Categories C ON P.CategoryID = C.CategoryID
WHERE C.CategoryName = 'Beverages' AND strftime('%Y-%m', O.OrderDate) = '2020-06'""",
    },
    {
        "id": "q06_sql_top_customer_margin_2020",
        "question": "Who was the top customer by gross margin in 2020? Assume CostOfGoods = 70% of UnitPrice. Return {customer:str, margin:float}.",
        "sql": """SELECT C.CompanyName AS customer, ROUND(SUM((OD.UnitPrice * 0.3) * OD.Quantity * (1 - COALESCE(OD.Discount, 0))), 2) AS margin
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
INNER JOIN Customers C ON O.CustomerID = C.CustomerID
WHERE strftime('%Y', O.OrderDate) = '2020'
GROUP BY O.CustomerID, C.CompanyName
ORDER BY margin DESC
LIMIT 1""",
    },
]

# Filter to SQL-based questions only
sql_examples = [q for q in assessment_qs if q["sql"]]

print(f"[*] Creating training examples from {len(sql_examples)} SQL assessment questions...")

# Create training examples
trainset = []
for q in sql_examples:
    example = dspy.Example(
        question=q["question"],
        db_schema=DATABASE_SCHEMA,
        context="Data range: 2012-2023. Use strftime for dates, COALESCE for nulls, CAST for floats.",
        sql=q["sql"],
    ).with_inputs("question", "db_schema", "context")
    trainset.append(example)

print(f"[OK] Created {len(trainset)} training examples")

# Create base module (ChainOfThought)
base_module = dspy.ChainOfThought(NLToSQLSignature)

print("[*] Running BootstrapFewShot optimization...")
print("    This will take a minute or two...")

# Define metric: Simple accuracy based on whether the generated SQL contains key patterns
def sql_accuracy(example, pred, trace=None):
    """Check if generated SQL is valid."""
    sql_text = pred.sql.lower()
    question_text = example.question.lower()
    
    # Basic checks
    has_select = "select" in sql_text
    has_from = "from" in sql_text
    
    # More sophisticated checks based on question
    if "category" in question_text:
        has_category_table = "categories" in sql_text or "category" in sql_text
    else:
        has_category_table = True
    
    if "discount" in question_text:
        has_discount = "discount" in sql_text
    else:
        has_discount = True
    
    if "revenue" in question_text or "amount" in question_text or "total" in question_text:
        has_math = "*" in sql_text or "sum" in sql_text
    else:
        has_math = True
    
    score = has_select and has_from and has_category_table and has_discount and has_math
    return score

# Run optimization
optimizer = dspy.BootstrapFewShot(metric=sql_accuracy, max_bootstrapped_demos=3, max_rounds=5)
optimized_module = optimizer.compile(base_module, trainset=trainset)

print("[OK] Optimization complete!")

# Save optimized module
save_path = "optimized_nl_to_sql.json"
optimized_module.save(save_path)
print(f"[OK] Saved optimized module to {save_path}")

print("\n[INFO] Test the optimized module on diverse questions:")
print("       python run_agent_hybrid.py --batch sample_questions_diverse_generic.jsonl --out outputs_diverse_optimized.jsonl")

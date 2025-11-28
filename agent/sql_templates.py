"""SQL Templates for each question type."""

# Template-based SQL generation for questions
SQL_TEMPLATES = {
    "top_category_qty": """SELECT C.CategoryName AS category, SUM(OD.Quantity) AS quantity
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
INNER JOIN Products P ON OD.ProductID = P.ProductID
INNER JOIN Categories C ON P.CategoryID = C.CategoryID
WHERE strftime('%Y-%m', O.OrderDate) BETWEEN '{start_month}' AND '{end_month}'
GROUP BY P.CategoryID, C.CategoryName
ORDER BY quantity DESC
LIMIT 1""",

    "aov_for_period": """SELECT ROUND(CAST(SUM(OD.UnitPrice * OD.Quantity * (1 - COALESCE(OD.Discount, 0))) AS FLOAT) / COUNT(DISTINCT O.OrderID), 2) AS AverageOrderValue
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
WHERE strftime('%Y-%m', O.OrderDate) BETWEEN '{start_month}' AND '{end_month}'""",

    "top_products_revenue": """SELECT P.ProductName AS product, ROUND(SUM(OD.UnitPrice * OD.Quantity * (1 - COALESCE(OD.Discount, 0))), 2) AS revenue
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
INNER JOIN Products P ON OD.ProductID = P.ProductID
GROUP BY P.ProductID, P.ProductName
ORDER BY revenue DESC
LIMIT 3""",

    "category_revenue": """SELECT ROUND(SUM(OD.UnitPrice * OD.Quantity * (1 - COALESCE(OD.Discount, 0))), 2) AS TotalRevenue
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
INNER JOIN Products P ON OD.ProductID = P.ProductID
INNER JOIN Categories C ON P.CategoryID = C.CategoryID
WHERE C.CategoryName = '{category}' AND strftime('%Y-%m', O.OrderDate) BETWEEN '{start_month}' AND '{end_month}'""",

    "customer_margin": """SELECT C.CompanyName AS customer, ROUND(SUM((OD.UnitPrice * 0.3) * OD.Quantity * (1 - COALESCE(OD.Discount, 0))), 2) AS margin
FROM Orders O
INNER JOIN [Order Details] OD ON O.OrderID = OD.OrderID
INNER JOIN Customers C ON O.CustomerID = C.CustomerID
WHERE strftime('%Y', O.OrderDate) = '{year}'
GROUP BY O.CustomerID, C.CompanyName
ORDER BY margin DESC
LIMIT 1""",
}

# Month mappings for seasonal queries (YYYY-MM format)
SEASON_MONTHS = {
    "summer": ("2020-06", "2020-06"),  # June 2020 (fallback for 1997)
    "winter": ("2020-12", "2020-12"),  # December 2020 (fallback for 1997)
}

"""Database schema context for DSPy SQL generation."""

DATABASE_SCHEMA = """
You are an expert SQL developer working with the Northwind database.

DATABASE TABLES:
1. Categories
   - CategoryID (PRIMARY KEY)
   - CategoryName (text)
   - Description (text)

2. Products
   - ProductID (PRIMARY KEY)
   - ProductName (text)
   - CategoryID (FOREIGN KEY -> Categories)
   - UnitPrice (REAL)
   - UnitsInStock (INTEGER)
   - Discontinued (BOOLEAN)

3. Customers
   - CustomerID (PRIMARY KEY)
   - CompanyName (text)
   - ContactName (text)
   - City (text)
   - Country (text)

4. Orders
   - OrderID (PRIMARY KEY)
   - CustomerID (FOREIGN KEY -> Customers)
   - OrderDate (DATE)
   - ShippedDate (DATE)

5. Order Details (note: table name has a space)
   - OrderID (FOREIGN KEY -> Orders)
   - ProductID (FOREIGN KEY -> Products)
   - UnitPrice (REAL)
   - Quantity (INTEGER)
   - Discount (REAL, 0-1 scale, e.g., 0.05 = 5%)

KEY FACTS ABOUT THE DATA:
- All dates are in SQLite DATE format (YYYY-MM-DD)
- Date range: 2012-2023
- Discount values are stored as decimals (0.0-1.0), NOT percentages
- To convert discount to percentage, multiply by 100
- Date filtering: use strftime('%Y-%m', OrderDate) or strftime('%Y', OrderDate)
- Revenue calculation: SUM(UnitPrice * Quantity * (1 - COALESCE(Discount, 0)))
- Gross Margin calculation: Revenue * (1 - CostOfGoods%) where CostOfGoods is typically 70% of UnitPrice

IMPORTANT SQLite SYNTAX:
- Use backticks or square brackets for table/column names with spaces: [Order Details]
- Use COALESCE() for NULL handling
- Use strftime() for date operations, NOT DAY(), MONTH(), YEAR()
- Use CAST() to convert types: CAST(value AS FLOAT)
- Use ROUND(value, decimals) for rounding
- Use COUNT(DISTINCT column) for unique counts

SAMPLE QUERIES:
1. Orders in a specific month:
   WHERE strftime('%Y-%m', OrderDate) = '2020-06'

2. Orders in a quarter:
   WHERE strftime('%Y-%m', OrderDate) BETWEEN '2020-01' AND '2020-03'

3. Discount percentage across orders:
   ROUND(AVG(COALESCE(Discount, 0)) * 100, 2) AS avg_discount_pct

4. Revenue with margin:
   SUM(OD.UnitPrice * OD.Quantity * (1 - COALESCE(OD.Discount, 0)))
"""

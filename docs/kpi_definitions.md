# KPI Definitions

## Average Order Value (AOV)
- **Formula:** AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)
- **Description:** Total revenue divided by number of distinct orders.

## Gross Margin
- **Formula:** GM = SUM((UnitPrice - CostOfGoods) * Quantity * (1 - Discount))
- **Note:** If CostOfGoods is missing, approximate with 70% of UnitPrice.
- **Calculation Method:** For Northwind DB, use CostOfGoods ≈ 0.7 * UnitPrice when cost data is unavailable.

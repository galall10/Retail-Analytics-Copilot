"""
SQLite database tool for Northwind queries.
Handles schema introspection and query execution.
"""

import sqlite3
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Result of a SQL query execution."""
    success: bool
    columns: list[str]
    rows: list[tuple]
    error: str = ""
    row_count: int = 0


class SQLiteTool:
    """Interface to Northwind SQLite database."""

    def __init__(self, db_path: str = "data/northwind.sqlite"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._ensure_connection()

    def _ensure_connection(self):
        """Ensure database connection is active."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to connect to {self.db_path}: {e}")

    def get_schema(self) -> dict[str, Any]:
        """
        Get database schema with table names, columns, and types.
        
        Returns:
            Dictionary mapping table_name -> {columns: [(name, type), ...]}
        """
        schema = {}
        cursor = self.conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info([{table_name}])")
            columns = cursor.fetchall()
            col_info = [(col[1], col[2]) for col in columns]  # name, type
            schema[table_name] = {"columns": col_info}
        
        cursor.close()
        return schema

    def execute_query(self, query: str, limit: Optional[int] = None) -> QueryResult:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            limit: Optional row limit (appended if not already present)
        
        Returns:
            QueryResult with success status, columns, rows, and any error message
        """
        try:
            # Add LIMIT if not present and limit is specified
            query_str = query.strip()
            if limit and "LIMIT" not in query_str.upper():
                query_str += f" LIMIT {limit}"
            
            cursor = self.conn.cursor()
            cursor.execute(query_str)
            
            # Fetch all results
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            cursor.close()
            
            return QueryResult(
                success=True,
                columns=columns,
                rows=rows,
                row_count=len(rows)
            )
        
        except sqlite3.Error as e:
            return QueryResult(
                success=False,
                columns=[],
                rows=[],
                error=str(e),
                row_count=0
            )
        except Exception as e:
            return QueryResult(
                success=False,
                columns=[],
                rows=[],
                error=f"Unexpected error: {str(e)}",
                row_count=0
            )

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Ensure connection is closed on cleanup."""
        self.close()

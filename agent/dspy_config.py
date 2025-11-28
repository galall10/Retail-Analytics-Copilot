"""
DSPy configuration and utilities for the retail analytics agent.
Handles LM setup, caching, and format validation.
"""

import json
import re
from typing import Any

import dspy


def configure_dspy(
    model: str = "phi3.5",
    api_base: str = "http://localhost:11434",
    temperature: float = 0.1,
    max_tokens: int = 500,
):
    """
    Configure DSPy with Ollama.
    
    Args:
        model: Ollama model name
        api_base: Ollama API base URL
        temperature: LM temperature (0.1 = deterministic)
        max_tokens: Max tokens per response
    """
    dspy.configure(
        lm=dspy.LM(
            f"ollama_chat/{model}",
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


def validate_answer_format(answer: str, format_hint: str) -> tuple[bool, Any]:
    """
    Validate that answer matches format_hint.
    
    Args:
        answer: Generated answer (as string)
        format_hint: Expected format ('int', 'float', 'str', or JSON schema)
    
    Returns:
        (is_valid, parsed_value)
    """
    if not answer:
        return False, None
    
    answer_str = str(answer).strip()
    
    # int format
    if format_hint == "int":
        try:
            match = re.search(r'-?\d+', answer_str)
            if match:
                val = int(match.group())
                return True, val
            return False, None
        except:
            return False, None
    
    # float format
    elif format_hint == "float":
        try:
            match = re.search(r'-?\d+\.?\d*', answer_str)
            if match:
                val = float(match.group())
                return True, round(val, 2)
            return False, None
        except:
            return False, None
    
    # dict/object format
    elif format_hint.startswith("{"):
        try:
            # Try JSON parse first
            parsed = json.loads(answer_str)
            return True, parsed
        except:
            # Try to extract from dict-like string
            try:
                inner = answer_str.strip("{}").strip()
                result = {}
                for pair in inner.split(","):
                    if ":" in pair:
                        key, val = pair.split(":", 1)
                        key = key.strip().strip("'\"")
                        val = val.strip().strip("'\"")
                        result[key] = val
                return bool(result), result if result else None
            except:
                return False, None
    
    # list format
    elif format_hint.startswith("list["):
        try:
            parsed = json.loads(answer_str)
            return isinstance(parsed, list), parsed
        except:
            return False, None
    
    # string format (default)
    else:
        return True, answer_str


def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON object from LLM response (handles markdown code blocks).
    
    Args:
        response: LLM response text
    
    Returns:
        Parsed JSON dict, or empty dict if not found
    """
    # Try to extract from ```json ... ```
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try to extract from ```  ... ```
    code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except:
            pass
    
    # Try direct JSON parse
    try:
        return json.loads(response)
    except:
        pass
    
    return {}


def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL from LLM response (handles markdown code blocks).
    
    Args:
        response: LLM response text
    
    Returns:
        SQL query string
    """
    # Extract from ```sql ... ```
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Extract from ``` ... ```
    code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Return as-is
    return response.strip()

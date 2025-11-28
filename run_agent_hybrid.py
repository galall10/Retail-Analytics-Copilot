"""
Main CLI entrypoint for the retail analytics copilot.
Uses DSPy 3.0 modules and LangGraph orchestration.

Usage:
    python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Any

import click
from pydantic import BaseModel, Field

from agent.graph_hybrid import SimpleHybridAgent
from agent.dspy_config import validate_answer_format


class OutputRow(BaseModel):
    """Output contract for each question."""
    id: str
    final_answer: Any
    sql: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = ""
    citations: list[str] = Field(default_factory=list)


@click.command()
@click.option(
    "--batch",
    type=click.Path(exists=True),
    required=True,
    help="Path to batch file (JSONL with questions)"
)
@click.option(
    "--out",
    type=click.Path(),
    required=True,
    help="Path to output file (will be created)"
)
def main(batch: str, out: str):
    """
    Run the retail analytics copilot on a batch of questions.
    """
    
    # Initialize agent
    print("[*] Initializing agent...", file=sys.stderr)
    agent = SimpleHybridAgent(db_path="data/northwind.sqlite", docs_dir="docs")
    
    # Load questions
    print(f"[*] Loading questions from {batch}", file=sys.stderr)
    questions = []
    with open(batch, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    print(f"[*] Loaded {len(questions)} questions", file=sys.stderr)
    
    # Process each question
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as out_f:
        for idx, q_obj in enumerate(questions, 1):
            q_id = q_obj.get("id", f"q{idx}")
            question = q_obj.get("question", "")
            format_hint = q_obj.get("format_hint", "str")
            
            print(f"[{idx}/{len(questions)}] Processing: {q_id}", file=sys.stderr)
            
            try:
                # Run agent
                state = agent.run(question, format_hint)
                
                # Validate and parse answer
                final_answer = state["final_answer"]
                is_valid, parsed_answer = validate_answer_format(final_answer, format_hint)
                
                if not is_valid:
                    # Try to parse as-is if format validation failed
                    print(f"  [!] Format mismatch: expected {format_hint}, got {type(final_answer)}", file=sys.stderr)
                    parsed_answer = final_answer
                
                # Build output row
                output = OutputRow(
                    id=q_id,
                    final_answer=parsed_answer,
                    sql=state.get("sql", ""),
                    confidence=state.get("confidence", 0.0),
                    explanation=state.get("explanation", "")[:200],
                    citations=state.get("citations", [])
                )
                
                # Write as JSON
                out_f.write(json.dumps(output.model_dump()) + "\n")
                out_f.flush()
                
                print(f"  ✓ Answer: {output.final_answer}", file=sys.stderr)
            
            except Exception as e:
                print(f"  ✗ Error: {e}", file=sys.stderr)
                # Write error row
                output = OutputRow(
                    id=q_id,
                    final_answer=None,
                    sql="",
                    confidence=0.0,
                    explanation=f"Error: {str(e)[:200]}",
                    citations=[]
                )
                out_f.write(json.dumps(output.model_dump()) + "\n")
                out_f.flush()
    
    print(f"\n[✓] Results written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

import logging
import os
import subprocess
import sys
from pathlib import Path

from typing import Annotated, Dict, List, Union
from enum import Enum

from fastmcp import FastMCP


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_TIMEOUT_SECONDS = 600
QUERY_TIMEOUT_SECONDS = int(
	os.getenv("GRAPHRAG_QUERY_TIMEOUT_SECONDS", str(DEFAULT_QUERY_TIMEOUT_SECONDS))
)

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("graphrag_mcp")

mcp = FastMCP("graphrag")


class QueryMode(str, Enum):
	LOCAL = "local"
	GLOBAL = "global"
	DRIFT = "drift"
	BASIC = "basic"
	

@mcp.prompt()
def graphrag_query_prompt(question: str, mode: QueryMode = QueryMode.LOCAL) -> str:
	"""Prompt that lets the caller explicitly choose the type of GraphRAG query."""
	return (
		"Choose the type of GraphRAG query to run based on your question's needs:\n"
		"- `local`: For questions about specific entities or details in the documents.\n"
		"- `global`: For questions about the dataset as a whole.\n"
		"- `drift`: For comprehensive local search using community insights.\n"
		"- `basic`: For basic vector RAG search, useful for comparing search types.\n"
		f"\nSelected mode: {mode.value}\n"
		f"Question: {question}"
	)


def _run_graphrag_query(question: str, method: QueryMode) -> Dict[str, str]:
	if not question.strip():
		raise ValueError("Question is required")

	args: List[str] = [sys.executable, "-m", "graphrag", "query", "--method", method.value]
	args.append(question)

	LOGGER.info(
		"Running GraphRAG query | method=%s | timeout_seconds=%s | command=%s",
		method,
		QUERY_TIMEOUT_SECONDS,
		" ".join(args[:-1]+ [f'"{args[-1]}"']),
	)
	timeout = None if QUERY_TIMEOUT_SECONDS <= 0 else QUERY_TIMEOUT_SECONDS
	result = None
	try:
		LOGGER.info("Subprocess call: %s", " ".join(args))
		LOGGER.info("Subprocess cwd: %s", str(BASE_DIR))
		LOGGER.info("Subprocess env: %s", {k: v for k, v in os.environ.items() if k.startswith('AZURE') or k.startswith('OPENAI')})
		result = subprocess.run(
			args,
			cwd=str(BASE_DIR),
			capture_output=True,
			text=True,
			check=False,
			timeout=timeout,
			env=os.environ.copy(),
			stdin=subprocess.DEVNULL,
		)
	except subprocess.TimeoutExpired as e:
		LOGGER.error(
			"GraphRAG query timed out | timeout_seconds=%s | partial stdout=%r | partial stderr=%r",
			QUERY_TIMEOUT_SECONDS,
			getattr(e, 'stdout', None),
			getattr(e, 'stderr', None),
		)
		return {
			"stdout": getattr(e, 'stdout', ''),
			"stderr": (getattr(e, 'stderr', '') + " - GraphRAG query timed out"),
			"returncode": "-1",
		}

	LOGGER.info(
		"GraphRAG query completed | returncode=%s | stdout_len=%s | stderr_len=%s",
		result.returncode,
		len(result.stdout or ""),
		len(result.stderr or ""),
	)

	return {
		"stdout": result.stdout.strip(),
		"stderr": result.stderr.strip(),
		"returncode": str(result.returncode),
	}




@mcp.tool()
def graphrag_query(
	question: Annotated[str, "User question to run as a query"],
	mode: Annotated[QueryMode, "Query mode: local, global, drift, or basic"] = QueryMode.LOCAL,
) -> Dict[str, str]:
	"""Run a GraphRAG query against the indexed graph data with the specified mode."""
	if not isinstance(mode, QueryMode):
		try:
			mode = QueryMode(mode)
		except Exception:
			raise ValueError(f"Invalid query mode: {mode}. Must be one of: {[m.value for m in QueryMode]}")
	return _run_graphrag_query(question, method=mode)


if __name__ == "__main__":
	mcp.run()

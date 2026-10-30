# Harness Engineering: Build an AI Data Analyst You Can Actually Trust

![GitHub](https://img.shields.io/github/license/DataForScience/Harness_Engineering)
[![Twitter @data4sci](https://img.shields.io/twitter/follow/data4sci)](https://twitter.com/intent/follow?screen_name=data4sci)
![GitHub top language](https://img.shields.io/github/languages/top/DataForScience/Harness_Engineering)
![GitHub repo size](https://img.shields.io/github/repo-size/DataForScience/Harness_Engineering)
![GitHub last commit](https://img.shields.io/github/last-commit/DataForScience/Harness_Engineering)

[![Data For Science](https://img.shields.io/badge/Data_For_Science-Subscribe-blue)](https://data4sci.substack.com/)
[![Data Science Briefing](https://img.shields.io/badge/Data_Science_Briefing-Subscribe-blue)](https://data4sci.com/newsletter)

### Code and slides to accompany the online workshop [Harness Engineering: Build an AI Data Analyst You Can Actually Trust](https://luma.com/51oqj4cn) by Data For Science.


We build an AI data analyst in Python and engineer the **harness** around the model: permissioned MCP tools, context management within a token budget, layered guardrails against prompt injection, verification of every number in a report, and fault-tolerant run supervision with retries, budgets and structured traces. No agent framework: you build the harness yourself, one module per notebook, into a small Python package (`harness/`). Notebooks 02-06 write `permissions.py`, `context.py`, `guardrails.py`, `verify.py`, `supervisor.py` and `analyst.py` with `%%writefile` cells; the agent loop, MCP client, mock model and data generator ship with the repo. Finished copies of every module are in `reference/`: if you skip a notebook, the next one restores what it needs, and `restore("<module>")` replaces a module you broke.

Everything runs on a laptop, with no GPU or cloud infrastructure: a real third-party MCP server ([`mcp-server-motherduck`](https://github.com/motherduckdb/mcp-server-motherduck), for DuckDB) and a local database generated on first use. The notebooks drive the analyst with Claude, which needs an Anthropic API key. To work fully offline and for free, switch any notebook to the deterministic mock model (see [Choosing the model](#choosing-the-model)).

The workshop slides are in [`slides/`](./slides) (PDF).

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies, set your Anthropic API key and launch Jupyter:

```bash
uv sync
export ANTHROPIC_API_KEY=...
uv run jupyter notebook
```

Run the notebooks from the repository root. To run the fault-injected test suite against the harness you built (missing modules are filled in from `reference/`; it also checks every exercise solution, the Claude adapter against a fake client, and that the notebooks still rebuild `reference/` exactly):

```bash
uv run pytest
```

The test suite runs fully offline and needs no API key: it always uses the mock model.

Each notebook ends with a 5-10 minute exercise that checks itself. The solution is in a collapsible **Show solution** block below the exercise (no extension needed, so it works in Notebook 7, JupyterLab, VS Code and GitHub), and in `solutions/`, which you can load into a cell with `%load solutions/<file>.py`.

## Choosing the model

Every notebook starts with a cell that chooses the model that drives the analyst:

```python
BACKEND = "claude"   # or "mock"
set_backend(BACKEND)
```

- `"mock"`: `MockAnalyst`, a deterministic, rule-based stand-in. Free, needs no API key, and reproduces exactly the failures described in the notebooks. Its failure modes (`fabricate`, `sloppy_units`, `greedy`, `gullible`, `stubborn`) can be switched on and off. Use it to work offline.
- `"claude"` (selected in every notebook): `ClaudeLLM` (in `harness/claude.py`), Claude Opus 5.5 through the Anthropic Messages API. It has the same `complete(messages, tools)` interface as the mock, so the harness doesn't change. It needs `ANTHROPIC_API_KEY` in your environment and costs a little per run. Claude decides for itself, so your outputs will differ from the ones described in the text, and it is much less likely to fall into the planted traps. The harness must hold either way: compare what each layer catches.

Options passed to `set_backend` go to `ClaudeLLM`, for example `set_backend("claude", model="claude-sonnet-5-5", effort="high")`. The fault matrix in notebook 06 always uses the mock, whatever the backend.

## Repository layout

| Path | Contents |
|:--|:--|
| `01 - ...ipynb` to `06 - ...ipynb` | The six workshop notebooks |
| `harness/` | The framework-free harness package. Ships with the agent loop and tool box (`agent.py`), an SDK-free MCP client (`mcp_client.py`), the model interface and mock (`llm.py`), the Claude adapter (`claude.py`), the dataset generator (`data.py`), the exercise checkers (`exercises.py`) and notebook helpers (`workshop.py`). The notebooks add `permissions.py`, `context.py`, `guardrails.py`, `verify.py`, `supervisor.py` and `analyst.py` |
| `reference/` | Finished copies of the modules the notebooks build |
| `solutions/` | Exercise solutions, one file per notebook |
| `tests/` | Fault-injection, guardrail, solution, Claude-adapter and notebook-sync tests |
| `slides/` | Workshop slides (PDF) |
| `data/` | Logos and images; the DuckDB database `trailhead.duckdb` is generated here on first use (`uv run python -m harness.data` rebuilds it) |
| `outputs/` | Created at run time for memory files and traces (git-ignored) |

## Schedule

| Notebook | Topic | 
|:--|:--|
| [01 - The AI Data Analyst](./01%20-%20The%20AI%20Data%20Analyst.ipynb) | The data, the DuckDB MCP server, the mock model and a naive agent loop that fails in four ways | 
| [02 - MCP Tools with Permissions](./02%20-%20MCP%20Tools%20with%20Permissions.ipynb) | Tool discovery, allowlists, argument validation, SQL classification, allow/ask/deny rules, approvals and audit logs |
| [03 - Context Within a Token Budget](./03%20-%20Context%20Within%20a%20Token%20Budget.ipynb) | Compact result summaries, expandable handles, conversation compaction and tiered memory | 
| [04 - Layered Guardrails](./04%20-%20Layered%20Guardrails.ipynb) | Indirect prompt injection, spotlighting and quarantine, SQL inspection, a locked-down database engine | 
| [05 - Verifying AI Reports](./05%20-%20Verifying%20AI%20Reports.ipynb) | Claims backed by re-runnable queries, unit and coverage checks, LLM-as-judge for clarity only | 
| [06 - Failure Handling and Fault Injection](./06%20-%20Failure%20Handling%20and%20Fault%20Injection.ipynb) | Timeouts, retries, re-planning, budgets, structured traces and a fault-injected CI test suite | 

The four failure scenarios we fix along the way: a fabricated explanation for a revenue drop, an oversized tool result, malicious instructions hidden in customer feedback, and questions about periods with missing data.

## Author

<table border="0">
 <tr>
	<td>
	  <img src="data/bgoncalves.png" alt="Bruno Gonçalves" width="150" height="150" style="border-radius: 50%; object-fit: cover;">
	</td>
	<td>
	  <h2>Bruno Gonçalves</h2>
	  <h3>Data For Science, Inc.</h3>
	  <p>
			Web: <a href="http://www.data4sci.com/">www.data4sci.com</a><br>
			Twitter/X: <a href="https://twitter.com/bgoncalves">@bgoncalves</a><br>
			LinkedIn: <a href="https://www.linkedin.com/in/bmtgoncalves/">@bmtgoncalves</a><br>
			Email: <a href="info@data4sci.com">info@data4sci.com</a><br>
			Schedule a Call: <a href="https://data4sci.com/call">https://data4sci.com/call</a>
	  </p>
	</td>
 </tr>
</table>

# Ethicore Guardian — AI Threat Screening

Runtime AI-threat screening for Claude. This plugin adds the Ethicore Engine® Guardian MCP
server — four on-demand scans exposed as tools — plus a skill that tells Claude when to use them,
so Claude can screen untrusted input, tool calls, tool output, and documents **before it acts**.

## What it does

- **`analyze`** — screen a prompt, document, or web text before acting on it.
- **`scan_tool_call`** — vet a tool / function call the model chose before it executes.
- **`scan_tool_output`** — catch indirect prompt injection in a tool / API / RAG result before it re-enters context.
- **`scan_documents`** — vet ingested files (PDF/DOCX/PPTX/XLSX/RTF/TXT).

Each returns a **BLOCK / CHALLENGE / ALLOW** verdict with threat categories and reasoning, from a
consensus of pattern, semantic, behavioral, and ML layers across 180+ threat categories (prompt
injection, jailbreaks, data/secret exfiltration, tool-call abuse, RAG/memory poisoning, and more).

## Use it

Set your Guardian API key when you add the plugin. Then, in **Claude Code** (or a **local Cowork**
session), Claude uses the Guardian tools to screen untrusted prompts, tool calls, tool output, and
documents before taking action — blocking or challenging the dangerous ones.

## Paid feature

Requires a **Pro-plan-or-higher** Guardian API key; free/community keys are refused. Get one at
<https://portal.oraclestechnologies.com>. (Free access to Guardian remains available directly via the
API/SDK — this plugin is the paid, agent-native convenience.)

## Data

The plugin runs the published `ethicore-guardian-mcp` server (from PyPI) locally via `uvx`. The scans
you invoke send the text, tool arguments, tool output, or documents you pass them to the Guardian API
over TLS for threat analysis, authenticated with your API key. Payloads are analyzed transiently;
Guardian does not retain them as training data, and does not send your data anywhere other than the
Guardian API. Full privacy policy: <https://oraclestechnologies.com/legal/privacy>.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) installed (provides `uvx`), Python 3.10+.
- Works in **Claude Code** and **local Cowork** sessions. Local MCP servers do not load in claude.ai
  chat — for an everywhere-available connector, use the hosted Guardian MCP connector instead.
- Self-hosted (zero-egress) users run the license-delivered engine directly; this plugin targets the
  hosted edition.

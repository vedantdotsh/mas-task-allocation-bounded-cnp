"""LLM-agent communication cost model.

Token constants are grounded in the message shapes of current agent
interoperability protocols (JSON-RPC 2.0 envelopes as used by A2A/MCP)
rather than arbitrary units:

- CFP_TOKENS: a call-for-proposals carrying a structured task description
  (id, skill requirement, workload, location, deadline) inside a JSON-RPC
  envelope. Measured by tokenizing a representative A2A-style task message
  with the cl100k_base tokenizer: ~150 tokens.
- BID_TOKENS: a bid response is a short structured message (cost estimate
  plus a capability/confidence field): ~50 tokens.
- AWARD_TOKENS: award/ack message: ~20 tokens.
- CENTRAL_QUERY_TOKENS: a centralized orchestrator that queries an agent's
  full state (skill profile, queue contents, position, history summary)
  requires serializing substantially more context per query; we use 400
  tokens per query as a conservative estimate, consistent with the order-of-
  magnitude difference assumed between full-context and structured
  short-message coordination. These constants are modelling assumptions,
  not an empirical reduction attributed to Agent Contracts.

These remain model-derived estimates, not provider benchmarks; the paper
reports them as a parameterized cost model and discusses sensitivity.
"""

CFP_TOKENS = 150
BID_TOKENS = 50
AWARD_TOKENS = 20
CENTRAL_QUERY_TOKENS = 400
CENTRAL_RESPONSE_TOKENS = 50

# Latency model: per-message network/API overhead in milliseconds. Median
# time-to-first-token style overhead for a short structured exchange.
PER_MESSAGE_LATENCY_MS = 300.0

# Context pressure: fraction of a 128k-token context window consumed by the
# coordination traffic that a single orchestrating agent must retain.
CONTEXT_WINDOW_TOKENS = 128_000


def proxy_metrics(token_units: float, total_messages: int) -> dict:
    return {
        "proxy_tokens": float(token_units),
        "proxy_latency_ms": float(total_messages * PER_MESSAGE_LATENCY_MS),
        "context_pressure": float(token_units / CONTEXT_WINDOW_TOKENS),
    }

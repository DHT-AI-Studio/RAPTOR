from prometheus_client import Counter, Histogram

guardrails_checks_total = Counter(
    "guardrails_checks_total",
    "Total guardrail checks performed",
    ["check_type", "result", "mode"],
)

guardrails_violations_total = Counter(
    "guardrails_violations_total",
    "Total violations detected by guardrails",
    ["check_type", "category"],
)

guardrails_check_duration_seconds = Histogram(
    "guardrails_check_duration_seconds",
    "Time spent performing a guardrail content check",
    ["check_type"],
)

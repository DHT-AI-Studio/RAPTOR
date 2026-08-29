"""Search-space sampling + clamping (AUTOTUNE Phase B).

Two responsibilities, both pure functions (no I/O, easy to test):

- ``sample_random`` — draw one config point from a search space. This is the
  Step-1 "proposer": a random search that validates the whole loop before the
  smolagents agent is wired in.
- ``clamp_to_search_space`` — the code-side safety whitelist (AUTOTUNE §B2/§8).
  Whatever proposes a config (random or agent), its output is forced back inside
  the declared ranges / choices before it ever reaches training. Keys outside
  the search space are dropped.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Mapping

from app.models.experiment import SearchDimension


def _coerce_int(value: float) -> int:
    return int(round(value))


def sample_random(search_space: Mapping[str, SearchDimension],
                  rng: random.Random | None = None) -> Dict[str, Any]:
    """Draw one random config point, each value inside its dimension's range."""
    r = rng or random
    config: Dict[str, Any] = {}
    for name, dim in search_space.items():
        if dim.type == "categorical":
            config[name] = r.choice(list(dim.choices))
        elif dim.type in ("float", "int"):
            lo, hi = float(dim.min), float(dim.max)
            if dim.log:
                # Uniform in log space, then exponentiate.
                val = math.exp(r.uniform(math.log(lo), math.log(hi)))
            else:
                val = r.uniform(lo, hi)
            config[name] = _coerce_int(val) if dim.type == "int" else val
    return config


def _clamp_number(value: Any, dim: SearchDimension) -> Any:
    """Clamp a numeric value into [min, max], coercing type; None-safe."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        # Un-parseable → fall back to the range midpoint (safe default).
        num = (float(dim.min) + float(dim.max)) / 2.0
    num = max(float(dim.min), min(float(dim.max), num))
    return _coerce_int(num) if dim.type == "int" else num


def clamp_to_search_space(config: Mapping[str, Any],
                          search_space: Mapping[str, SearchDimension]) -> Dict[str, Any]:
    """Force a proposed config back inside the search space (code whitelist).

    - Numeric knobs are clamped into [min, max] (and int-coerced).
    - Categorical knobs are snapped to the nearest valid choice; anything
      invalid falls back to the first choice.
    - Keys not in the search space are dropped entirely (the agent may only
      touch declared knobs).
    """
    out: Dict[str, Any] = {}
    for name, dim in search_space.items():
        if name not in config:
            continue
        value = config[name]
        if dim.type == "categorical":
            out[name] = value if value in dim.choices else dim.choices[0]
        else:
            out[name] = _clamp_number(value, dim)
    return out

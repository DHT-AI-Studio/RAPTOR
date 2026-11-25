"""Guard-model adapters. Importing this package triggers self-registration
of every built-in adapter (see app/adapters/registry.py). Adding a new guard
model = a new file under app/adapters/models/ + one import line here.
"""
from app.adapters.models import (  # noqa: F401
    gpt_oss_safeguard,
    granite_guardian,
    llama_guard3,
)

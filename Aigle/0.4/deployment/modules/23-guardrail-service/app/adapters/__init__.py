"""Guard-model adapters. Importing this package triggers self-registration
of every built-in adapter (see app/adapters/registry.py). Adding a new guard
model = a new file under app/adapters/models/ + one import line here.
"""
from app.adapters.models import granite_guardian, gpt_oss_safeguard, llama_guard3  # noqa: F401

"""Auto-tuning: NL-driven fine-tune → evaluate optimization loop (AUTOTUNE).

Groups the optimization machinery that sits on top of the benchmark core:
- orchestrator    : the hard-budget control loop (Phase B)
- experiment_store: experiments-table persistence
- search_space    : sampling + clamp whitelist
- training_client : Module 16 training submit/poll/cancel
"""

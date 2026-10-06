# Contributing to RAPTOR

First off, thank you for considering contributing to RAPTOR! It's people like you that make RAPTOR such a great tool for the AI community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '....'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python version: [e.g., 3.9.7]
- RAPTOR version: [e.g., Aigle 0.1]

**Additional context**
Add any other context, logs, or screenshots about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

**Feature Request Template:**
```markdown
**Is your feature request related to a problem?**
A clear description of the problem. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context, mockups, or examples about the feature request.

**Would you like to implement this feature?**
Let us know if you're interested in implementing it yourself!
```

### Contributing Code

We love code contributions! Here are ways you can contribute:

- **Fix bugs**: Look for issues labeled `bug` or `good first issue`
- **Add features**: Check issues labeled `enhancement` or `feature-request`
- **Improve documentation**: Help make our docs better
- **Write tests**: Improve our test coverage
- **Optimize performance**: Make RAPTOR faster and more efficient

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/RAPTOR.git
cd RAPTOR

# Add upstream remote
git remote add upstream https://github.com/DHT-AI-Studio/RAPTOR.git
```

### 2. Find Your Way Around

Each release lives in its own folder under `Aigle/`. **New contributions should target the latest release folder** (currently `Aigle/0.4/`) unless the issue says otherwise.

```
Aigle/0.4/
├── deploy.sh                  # deploy / stop / status helper
├── deployment/modules/
│   ├── build.py               # module registry and build orchestration
│   ├── .env.example           # configuration template (placeholders only)
│   └── <id>-<name>/           # one directory per module, e.g. 25-personal-db-service
│       ├── docker-compose.yml
│       ├── Dockerfile
│       ├── requirements.txt
│       └── tests/unit/
├── API_REFERENCE.md, MCP_REFERENCE.md, A2A_REFERENCE.md
└── BUILD.md                   # full build and deployment guide
```

See `Aigle/0.4/BUILD.md` for prerequisites (Docker, Docker Compose, NVIDIA drivers for GPU modules).

### 3. Set Up a Module for Development

```bash
cd Aigle/0.4/deployment/modules
cp .env.example .env                  # fill in your own local values — never commit .env

# Work on one module in a virtual environment
cd 25-personal-db-service             # example module
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest pytest-asyncio

# Or build and run modules with Docker
cd ..
python3 build.py -m 25 --build-only   # build only
bash ../../deploy.sh -m 25            # deploy selected module(s)
```

### 4. Create a Branch

```bash
git checkout main
git pull upstream main
git checkout -b feat/123-short-description   # 123 = the GitHub issue number
```

See [Branch Naming Convention](#branch-naming-convention).

### 5. Make Your Changes

Write your code following our [coding standards](#coding-standards) and the [module conventions](#module-conventions).

### 6. Test Your Changes

```bash
# Unit tests for the module you changed
cd Aigle/0.4/deployment/modules/<id>-<name>
python -m pytest tests/unit/ -v

# Lint
pip install ruff && ruff check .

# Validate the module's compose file resolves with .env.example
docker compose config --quiet
```

### 7. Commit and Push

```bash
git add <files>
git commit -m "feat(25): add temporal-only TKG retrieval"
git push origin feat/123-short-description
```

### 8. Open a Pull Request

Go to the [RAPTOR repository](https://github.com/DHT-AI-Studio/RAPTOR) and click "New Pull Request". Use `main` as the base branch.

## 🔄 Development Workflow

### How Changes Flow

```
main ──●──────────●──────────●──────▶  (always the latest state)
        \                   ↑
         └─ feat/123-…──────┘  squash-merged by a maintainer via PR
```

- `main` is the only long-lived branch and always reflects the latest state.
- Every change — from maintainers too — goes through a Pull Request.
- One issue = one branch = one PR. Small, focused PRs are reviewed fastest.
- Maintainers merge with **Squash and merge**, so your PR becomes a single commit on `main`.

### Branch Naming Convention

```
<type>/<issue-number>-<short-description>
```

| Type | Use for | Example |
|---|---|---|
| `feat` | New feature | `feat/123-graphrag-route` |
| `fix` | Bug fix | `fix/456-null-window-bound` |
| `docs` | Documentation only | `docs/789-mcp-reference` |
| `refactor` | Restructuring without behavior change | `refactor/321-orchestrator-plumbing` |
| `test` | Tests only | `test/654-asset-e2e` |
| `perf` | Performance improvement | `perf/987-warm-embedder` |
| `chore` | Config, cleanup, dependencies | `chore/111-env-example-order` |
| `ci` | CI/CD workflow changes | `ci/222-docker-build-gate` |

Lowercase, hyphen-separated, 2–5 words. Including the issue number links your branch to the discussion.

### Keep Your Branch Updated

If `main` moves while your PR is open, rebase rather than merging `main` into your branch:

```bash
git fetch upstream
git rebase upstream/main
# resolve any conflicts: edit files → git add <files> → git rebase --continue
git push --force-with-lease origin feat/123-short-description
```

Always use `--force-with-lease`, never `--force`, so you can't overwrite commits you haven't seen.

### Versions and Tags

| Tag / folder | Meaning |
|---|---|
| `Aigle/0.x/` | Source for release 0.x |
| `v0.x.0` | Final release |
| `v0.x.0-rc.N` | Release candidate |
| `v0.x.N` (N > 0) | Patch release for 0.x |

Tags are created only by maintainers and are never moved or deleted.

### Module Conventions

- Keep changes inside the module's own directory: `Aigle/0.4/deployment/modules/<id>-<name>/`.
- Changes to shared files (`build.py`, `.env.example`, `deploy.sh`) affect every module — submit them as their own small PR and say so in the title.
- **Adding a new module?** Open an issue first so maintainers can assign the module ID. A new module needs: a `build.py` entry, its variables in `.env.example`, a `README.md`, a `docker-compose.yml` that passes `docker compose config`, and a health endpoint.
- Modules **17, 19 and 20** (OpenSearch hybrid search, Neo4j, graph service) are **retired since 0.4** and kept only for rollback. Don't add features or new dependencies on them — use Module 25 (`/personal/search/*`) instead.
- Don't commit model weights, datasets or large media files.

### Secrets and Configuration

- **Never commit** `.env` files, credentials, API keys, tokens or private keys.
- `.env.example` contains **placeholders only**, e.g. `DB_PASSWORD=<your_db_password>`.
- Add every new environment variable to `.env.example`.
- Accidentally committed a secret? Don't just delete it in a new commit — it stays in Git history. Report it privately as described in [SECURITY.md](SECURITY.md) so it can be rotated.

## 💻 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Prefer double quotes for strings
- **Imports**: Organized in three groups (stdlib, third-party, local)

### Code Example

```python
"""Module docstring describing what this module does."""

import os
import sys
from typing import List, Optional

import numpy as np
import torch

from raptor.core import BaseClass
from raptor.utils import helper_function


class MyClass(BaseClass):
    """Class docstring describing the class.
    
    Attributes:
        attribute1: Description of attribute1.
        attribute2: Description of attribute2.
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        """Initialize MyClass.
        
        Args:
            param1: Description of param1.
            param2: Description of param2. Defaults to None.
        """
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def method_name(self, arg1: List[str]) -> bool:
        """Method description.
        
        Args:
            arg1: Description of arg1.
            
        Returns:
            Description of return value.
            
        Raises:
            ValueError: When arg1 is empty.
        """
        if not arg1:
            raise ValueError("arg1 cannot be empty")
        
        # Implementation
        return True
```

### Documentation Standards

- **All public modules, classes, and functions must have docstrings**
- Use [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints for function parameters and return values
- Add inline comments for complex logic

### Testing Standards

- **Write tests for all new features and bug fixes**
- Aim for at least 80% test coverage
- Use descriptive test names: `test_feature_under_specific_condition`
- Use pytest fixtures for common setup
- Mock external dependencies

```python
def test_my_function_returns_expected_value():
    """Test that my_function returns the correct value."""
    result = my_function(input_data)
    assert result == expected_value

def test_my_function_raises_on_invalid_input():
    """Test that my_function raises ValueError on invalid input."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

## 📝 Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

- **type** — `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`
- **scope** — the module ID(s) you changed, comma-separated (`25`, `04,13,25`), or `docs`, `build`, `ci`
- **subject** — imperative mood, lowercase, no trailing period, ≤ 72 characters

Because PRs are squash-merged, **your PR title becomes the commit message on `main`** — write it in this format.

### Examples

```
feat(25): add temporal-only TKG retrieval

Generic temporal questions with no named entity now return facts
from the TKG instead of an empty result.

Closes #123
```

```
fix(04,13,25): stop false-negative re-analysis on duplicate uploads

Fixes #456
```

### Commit Best Practices

- Use the imperative mood ("Add feature", not "Added feature")
- Separate subject from body with a blank line; wrap the body at 72 characters
- Reference issues in the footer (`Closes #123`, `Fixes #456`)
- Commit as often as you like on your branch — commits are squashed on merge

## 🔃 Pull Request Process

### Before Submitting

- [ ] Branch is rebased on the latest `main`
- [ ] Unit tests added or updated under `<module>/tests/unit/`, and they pass locally
- [ ] Module's `docker-compose.yml` still validates (`docker compose config --quiet`)
- [ ] New environment variables added to `.env.example` (placeholders only)
- [ ] README / API reference updated if behavior or endpoints changed
- [ ] No secrets, `.env` files, model weights or large binaries committed
- [ ] PR title follows the [commit format](#commit-message-format)
- [ ] Self-review of the diff completed

### Pull Request Template

The repository's [pull request template](../.github/pull_request_template.md) is filled in automatically when you open a PR. Please complete every section, especially **Related Issues**, **How Has This Been Tested?** and the module(s) touched.

### Review Process

1. **Checks**: Make sure your own tests and validation pass; maintainers may run additional checks.
2. **Code Review**: At least one maintainer reviews every PR.
3. **Feedback**: Push follow-up commits to the same branch; rebase if `main` has moved.
4. **Merge**: A maintainer squash-merges the PR and the branch is deleted.

### Response Times

- Initial review: Within 3-5 business days
- Follow-up reviews: Within 2-3 business days
- Simple fixes: May be merged within 24 hours

## 🐛 Issue Guidelines

### Labels

We use labels to categorize issues:

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested
- `wontfix` - This will not be worked on
- `duplicate` - This issue already exists
- `priority: high` - High priority issue
- `priority: low` - Low priority issue

### Issue Lifecycle

1. **Open**: Issue created
2. **Triaged**: Labeled and assigned priority
3. **In Progress**: Someone is working on it
4. **Review**: Pull request under review
5. **Closed**: Issue resolved or won't fix

## 🌐 Community

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **Telegram**: For community discussions (link coming soon)
- **Instagram**: For updates and showcases
- **X (Twitter)**: For announcements and news

### Getting Help

- Check the [documentation](docs/)
- Search existing [issues](https://github.com/DHT-AI-Studio/RAPTOR/issues)
- Ask questions in GitHub Discussions
- Join our Telegram group

### Recognition

We value all contributions! Contributors will be:

- Listed in our CONTRIBUTORS.md file
- Mentioned in release notes (for significant contributions)
- Featured on social media (with permission)

## 📞 Contact

For questions about contributing, contact the DHT Taiwan Team:

- **GitHub**: Open an issue or discussion
- **Company**: [DHT Solutions](https://dhtsolution.com/)

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to RAPTOR!** 🎉

Your efforts help make AI technology more accessible to everyone.

---

*Maintained by DHT Taiwan Team*


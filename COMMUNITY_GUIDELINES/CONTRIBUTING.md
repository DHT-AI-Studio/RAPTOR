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

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd Aigle/0.1
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install in editable mode
pip install -e .
```

### 3. Create a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 4. Make Your Changes

Write your code, following our [coding standards](#coding-standards).

### 5. Test Your Changes

```bash
# Run tests
pytest

# Run linting
flake8 .
pylint raptor/

# Run type checking (if using type hints)
mypy .

# Check test coverage
pytest --cov=raptor tests/
```

### 6. Commit and Push

```bash
# Stage your changes
git add .

# Commit with a meaningful message
git commit -m "Add feature: description of your feature"

# Push to your fork
git push origin feature/your-feature-name
```

### 7. Open a Pull Request

Go to the [RAPTOR repository](https://github.com/DHT-AI-Studio/RAPTOR) and click "New Pull Request".

## 🔄 Development Workflow

### Branch Naming Convention

- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/what-changed` - Documentation updates
- `refactor/what-refactored` - Code refactoring
- `test/what-tested` - Test additions or modifications
- `perf/what-optimized` - Performance improvements

### Keep Your Branch Updated

```bash
# Fetch latest changes from upstream
git fetch upstream

# Rebase your branch on upstream/main
git rebase upstream/main

# Force push to your fork (if already pushed)
git push --force-with-lease origin feature/your-feature-name
```

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

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

### Examples

```
feat(core): add support for custom model loading

Implement new ModelLoader class that supports loading custom
AI models from various sources including local files and remote URLs.

Closes #123
```

```
fix(utils): resolve memory leak in data preprocessing

Fixed memory leak caused by unreleased resources in the
preprocess_data function. Added proper cleanup in finally block.

Fixes #456
```

### Commit Best Practices

- Use the imperative mood ("Add feature" not "Added feature")
- Keep the subject line under 50 characters
- Separate subject from body with a blank line
- Wrap body at 72 characters
- Reference issues and pull requests in the footer

## 🔃 Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated (if needed)
- [ ] No merge conflicts with main branch
- [ ] Commit messages follow guidelines
- [ ] Self-review of code completed

### Pull Request Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Related Issues
Closes #(issue number)

## How Has This Been Tested?
Describe the tests that you ran to verify your changes.

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)

## Additional Notes
```

### Review Process

1. **Automated Checks**: CI/CD will run tests and linting
2. **Code Review**: At least one maintainer will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, a maintainer will merge your PR

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


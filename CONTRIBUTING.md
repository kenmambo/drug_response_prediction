# Contributing to Drug Response Prediction

Thank you for considering contributing to the Drug Response Prediction project! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your feature or bugfix

## Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/drug_response_prediction.git
cd drug_response_prediction

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS

# Install dependencies
pip install -e .
```

## Code Style

This project follows PEP 8 style guidelines. Please ensure your code adheres to these standards.

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the documentation if necessary
3. The PR should work for Python 3.8 and above
4. Ensure all tests pass before submitting the PR
5. Include a clear description of the changes in your PR

## Testing

Before submitting a pull request, please run the tests to ensure your changes don't break existing functionality:

```bash
pytest
```

## Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)

## Feature Requests

Feature requests are welcome. Please provide:

- A clear description of the feature
- The motivation for the feature
- Any potential implementation details you have in mind

## Code of Conduct

Please be respectful and inclusive in your interactions with other contributors. We aim to foster an open and welcoming environment for everyone.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
# Contributing to DeepAgents

Thank you for your interest in contributing to DeepAgents! This playground welcomes contributions from the community.

## How to Contribute

### Adding New Examples

We're always looking for more examples! Consider adding:

1. **LangChain Examples**: New agent types, different tools, or interesting use cases
2. **LangGraph Examples**: Complex workflows, advanced state management, or real-world scenarios
3. **Deep Agents**: Novel agent architectures, collaborative systems, or reasoning approaches

### Example Structure

When adding a new example, follow this structure:

```python
"""
Brief description of the example.

This example demonstrates [what it does].
"""

import os
from dotenv import load_dotenv
# ... other imports

# Load environment variables
load_dotenv()


def main():
    """Run the example."""
    print("=== Example Name ===\n")
    # Your code here


if __name__ == "__main__":
    main()
```

### File Naming Convention

- Use descriptive names with numbers: `01_basic_example.py`, `02_advanced_example.py`
- Place examples in the appropriate directory:
  - `examples/langchain/` for LangChain examples
  - `examples/langgraph/` for LangGraph examples
  - `examples/deepagents/` for deep agent examples

### Code Style

- Use clear, descriptive variable names
- Add docstrings to functions and classes
- Include comments for complex logic
- Follow PEP 8 style guidelines
- Run `black .` for formatting (if you have dev dependencies installed)

### Documentation

When adding a new example:

1. Add a clear docstring at the top of the file
2. Update the README.md with the new example
3. Update `run_examples.py` if adding a numbered example
4. Add comments explaining key concepts

### Testing Your Changes

Before submitting:

1. Run your example to ensure it works
2. Check for syntax errors: `python -m py_compile your_file.py`
3. Verify all imports are in `requirements.txt`
4. Test with fresh API keys to ensure setup instructions work

## Types of Contributions

### Examples
Add new examples demonstrating different concepts or use cases.

### Documentation
Improve README, add tutorials, or clarify existing documentation.

### Bug Fixes
Fix issues in existing examples or improve error handling.

### Enhancements
Improve existing examples with better practices or additional features.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-example`
3. Add your changes
4. Test thoroughly
5. Commit with clear messages: `git commit -m "Add example for X"`
6. Push to your fork: `git push origin feature/my-new-example`
7. Create a Pull Request

## Pull Request Guidelines

- Provide a clear description of what your PR does
- Reference any related issues
- Include examples of the output (if applicable)
- Ensure all examples run without errors
- Keep changes focused and minimal

## Questions?

Feel free to open an issue for:
- Questions about contributing
- Suggestions for new examples
- Discussion about architecture or best practices

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on the code, not the person
- Collaborate openly and transparently

Thank you for making DeepAgents better! 🙏

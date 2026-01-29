# Contributing to IBM AI Engineer Specialization

First off, thank you for considering contributing to this repository! 🎉

This document provides guidelines for contributing to the IBM AI Engineer Specialization repository. Following these guidelines helps maintain the quality and consistency of the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Contribution Guidelines](#contribution-guidelines)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful and considerate of others
- Welcome newcomers and encourage diverse perspectives
- Focus on constructive feedback
- Accept responsibility and apologize for mistakes

## How Can I Contribute?

There are many ways to contribute to this repository:

### 🐛 Reporting Bugs

If you find a bug or error in the notebooks:

1. Check if the issue has already been reported
2. Create a new issue with a clear title and description
3. Include steps to reproduce the bug
4. Provide information about your environment (Python version, OS, etc.)

### 💡 Suggesting Enhancements

Have an idea for improvement?

1. Check if the enhancement has already been suggested
2. Create a new issue describing your enhancement
3. Explain why this enhancement would be useful
4. Provide examples if possible

### 📝 Improving Documentation

Documentation improvements are always welcome:

- Fix typos or grammatical errors
- Clarify confusing explanations
- Add missing information
- Improve code comments
- Create or enhance examples

### 🔧 Contributing Code

You can contribute by:

- Adding new examples or notebooks
- Fixing bugs in existing code
- Improving code efficiency or readability
- Adding unit tests (if applicable)

## Getting Started

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/IBM-AI-Engineer-Specialization.git
   cd IBM-AI-Engineer-Specialization
   ```
3. **Create a new branch** for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up your environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt  # If available
   ```

## Contribution Guidelines

### For Jupyter Notebooks

- **Clear outputs before committing**: Run "Restart & Clear Output" before committing
- **Test your notebook**: Ensure all cells run without errors
- **Add markdown cells**: Include explanations and context
- **Use meaningful variable names**: Make code self-documenting
- **Include visualizations**: Add plots and graphs where appropriate
- **Add comments**: Explain complex logic or algorithms

### For Python Scripts

- **Follow PEP 8**: Use Python's style guide for code formatting
- **Add docstrings**: Document functions and classes
- **Handle exceptions**: Include proper error handling
- **Write tests**: Add unit tests for new functionality (if applicable)

### For Documentation

- **Use proper Markdown**: Follow Markdown best practices
- **Check links**: Ensure all hyperlinks work correctly
- **Be clear and concise**: Write in simple, understandable language
- **Use proper formatting**: Apply headers, lists, and code blocks appropriately

## Style Guidelines

### Python Code Style

```python
# Good: Clear, descriptive names and proper formatting
def calculate_mean_squared_error(predictions, actual_values):
    """
    Calculate the Mean Squared Error between predictions and actual values.
    
    Args:
        predictions (array): Predicted values
        actual_values (array): Ground truth values
    
    Returns:
        float: Mean squared error
    """
    squared_errors = (predictions - actual_values) ** 2
    return squared_errors.mean()

# Bad: Unclear names and poor formatting
def calc(p,a):
    return ((p-a)**2).mean()
```

### Markdown Style

```markdown
# Good: Clear hierarchy and formatting

## Section Title

Brief introduction to the section.

### Subsection

- **Key Point 1**: Description
- **Key Point 2**: Description

```python
# Code example with syntax highlighting
import numpy as np
```

# Bad: Inconsistent formatting and unclear structure

Section Title
some text
* point 1
* point 2
```

## Commit Messages

Write clear, meaningful commit messages:

### Good Examples

```
Add logistic regression example with visualization
Fix typo in neural networks documentation
Update requirements with TensorFlow 2.x
Improve performance of k-means clustering implementation
```

### Format

```
<type>: <subject>

<optional body>

<optional footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting changes (no code change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

## Pull Request Process

1. **Update documentation**: If you've changed functionality, update the relevant README files

2. **Test your changes**: Ensure all notebooks run without errors

3. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Describe what changes you've made and why
   - Reference any related issues
   - Include screenshots for visual changes

4. **Wait for review**: A maintainer will review your PR and may request changes

5. **Make requested changes**: Address any feedback from reviewers

6. **Merge**: Once approved, your PR will be merged

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have tested my changes
- [ ] I have updated the documentation
- [ ] All notebooks run without errors
- [ ] I have cleared outputs from notebooks
```

## Questions?

If you have questions or need help:

- Open an issue with the `question` label
- Check existing issues for similar questions
- Review the course materials and documentation

---

Thank you for contributing! Your efforts help make this resource better for everyone. 🙏

Happy coding! 🚀

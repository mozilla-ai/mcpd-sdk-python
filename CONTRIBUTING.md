# Contributing to Mozilla.ai `mcpd-sdk-python`

Thank you for your interest in contributing to the `mcpd` project! This project supports the mozilla.ai goal of empowering developers to integrate AI capabilities into their projects using open-source tools and models.

We welcome all kinds of contributions, from improving customization, to extending capabilities, to fixing bugs. Whether you’re an experienced developer or just starting out, your support is highly appreciated.

By contributing to this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## **How to Contribute**

### **Customize for your use-case or Extend It** 🔧
- Fork this repo and customize it for your own use-case or even extend its capabilities.
- We'd love to see what you've built!

### **Browse Existing Issues** 🔍
- Check the Issues page to see if there are any tasks you'd like to tackle.
- Look for issues labeled **`good first issue`** if you're new to the project—they're a great place to start.

### **Report Issues** 🐛
- Found a bug? Open a Bug Report by clicking on 'New Issue'
- Provide as much detail as possible, including the steps to reproduce the issue and expected vs. actual behavior

### **Suggest Features** 🚀
- Have an idea for improving the project? Open a Feature Request by clicking on 'New Issue'
- Share why the feature is important and any alternative solutions you’ve considered.

### **Submit Pull Requests** 💻
- Fork the repository and create a new branch for your changes.
- Install [pre-commit](https://pre-commit.com/) to ensure the code is formatted and standardized correctly, by running `uv sync --group lint` and then `pre-commit install`.
- Ensure your branch is up-to-date with the main branch before submitting the PR
- Please follow the PR template, adding as much detail as possible, including how to test the changes

---

### **Guidelines for Contributions**

**Coding Standards**
- Follow PEP 8 for Python formatting.
- Use clear variable and function names and add comments to improve readability.

**Testing**
- Test changes locally to ensure functionality.
- Install the package using development dependencies before testing: `uv sync --group dev`

**Documentation**
- Update docs for changes to functionality and maintain consistency with existing docs.

---

## Security Vulnerabilities

If you discover a security vulnerability, please **DO NOT** open a public issue. Report it responsibly by following our [Security Policy](SECURITY.md).

---

## License

By contributing, you agree that your contributions will be licensed as described in [LICENSE](LICENSE.md).

# Agent guide

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing code. Consult [README.md](README.md) for project scope and commands, [pyproject.toml](pyproject.toml) for dependencies and lint/test configuration, and `src/cautious_chainsaw/` for the current implementation. Follow the checked-in structure; update documentation when behavior or commands change.

## Working agreement

- Handle routine implementation and smoke checks without unnecessary questions. Keep local CPU unit tests independent of downloaded models or datasets. Run GPU and Hugging Face integration checks in Colab. See [CONTRIBUTING.md](CONTRIBUTING.md) for the maintainer test environment and commands.
- Before running resource-heavy work or containers over SSH, follow the resource limits in [CONTRIBUTING.md](CONTRIBUTING.md).

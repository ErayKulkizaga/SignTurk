# Contributing

Thank you for your interest in contributing to this project!

## Reporting Issues

If you encounter a bug or unexpected behavior:

1. Check if the issue has already been reported in [Issues](https://github.com/ErayKulkizaga/SignTurk/issues)
2. If not, open a new issue with:
   - A clear title and description
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - Your environment (OS, Python version, browser)

## Suggesting Features

Open an issue with the `enhancement` label and describe:
- What the feature does
- Why it would be useful
- Any implementation ideas you have

## Pull Requests

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run the lightweight checks:
   - `python -m unittest discover -s tests -v`
   - `python -m text_processing.eval --check --min-exact 0.98`
   - `python -m compileall backend.py database.py models.py live_pipeline.py text_processing signturk_runtime research tools`
5. Commit with a clear message: `git commit -m "add: your feature description"`
6. Push and open a Pull Request against `main`

Do not commit model checkpoints or AUTSL-derived tensors. Publish versioned
project checkpoints as GitHub Release assets, record their size and SHA-256 in
`model-assets.json`, and verify them with `tools/download_models.py`.

## Areas Where Help Is Welcome

- Improving animation accuracy (better coordinate mapping)
- Improving recognition across signer, lighting, and background variation
- Mobile/responsive UI improvements
- Model performance improvements (new architectures, more data)
- Multilingual support (other sign languages)

## Code Style

- Python: follow PEP8, use descriptive variable names
- JavaScript: keep the browser UI dependency-light and accessible
- Keep functions small and focused

## Questions

Feel free to open an issue for any questions about the codebase.

# Changelog

## 2.0.3

- A Google Colab notebook, `colab/CollectiveLens.ipynb`, runs the whole app in the browser with nothing installed on your computer. It gets the code from GitHub (or from an uploaded zip), installs the requirements, reads your Anthropic key from Colab Secrets and starts the app.
- Tests for the notebook (valid, no saved outputs, no key, uses names that exist in the package).


## 2.0.2

- The `.env` file is now actually read. The README said the key could be put there, but the code only read real environment variables.
- A Mac launcher, `start-mac.command`, sets everything up with a double-click (Python check, environment, installation, optional key, browser).
- `COLLECTIVELENS_OPEN_BROWSER=1` opens the app in the browser when it is ready.


## 2.0.1

- A crash inside PyTorch (reported on a Mac with Python 3.15) closed the whole app. The app now runs a tiny PyTorch calculation in a separate process before loading the search model, and falls back to keyword matching, with a notice, if that crashes.
- The search model runs on the CPU by default (`COLLECTIVELENS_DEVICE`), the most reliable choice.
- README: Python 3.12 recommended, a troubleshooting section, and the new settings.


## 2.0.0

Version 1 is untouched in `legacy/` and tagged `v1.0-bootcamp`. The eight policy categories are the same as in version 1, so results stay comparable.

### Fixed

- The README of version 1 described BART, HDBSCAN, TF-IDF and UMAP, which were not in the code. The README now describes what the code does.
- Search now reads whole documents. Version 1 embedded each document as one block, but its model only reads the first part (about 128 tokens), so most of every document was ignored. Documents are now split into passages of about 180 words.
- Claude calls no longer break when a model returns a thinking block before its answer (version 1 read the first block only). Answers are read from the text blocks.
- Language detection is repeatable (fixed seed). Before, the same document could get a different language on different runs.
- A document that cannot be read is skipped with a message and no longer stops the whole build.
- The retired Claude models `claude-3-haiku-20240307` and `claude-3-5-haiku-20241022` were replaced. The default is now `claude-haiku-5-5`, changeable with `COLLECTIVELENS_CLAUDE_MODEL`.
- The app no longer creates a public link on launch (`share=True` in version 1).
- Summaries use up to 30,000 characters of each document instead of 8,000.

### Changed

- The "Avg Confidence 47%" figure was the mean cosine similarity between documents and category descriptions, which is not a measure of being right. It is now called similarity, documents between two topics are flagged, and `eval` measures agreement with topics chosen by a person.
- Search results show the best matching passages. The per-result "Why is this relevant?" call to Claude was removed (cheaper, faster, and the evidence is visible).
- Clusters are labelled with distinguishing words computed from the documents (KeyBERT was dropped). The number of groups is chosen automatically.
- The default search model is the multilingual `BAAI/bge-m3`. If it cannot be loaded, the app falls back to keyword matching and says so on screen.
- Settings come from environment variables and a categories file, so no code edit is needed.

### Added

- A new interface in Gradio 6: Search, Ask the library, Browse, Insights and About tabs, filters, a status line, dark mode and a layout that works on a phone.
- "Ask the library": answers written from retrieved passages only, with numbered citations that are checked.
- A demo library of 13 fictional documents in English, Italian and French, so the app runs without any private data.
- A command line (`build`, `eval`, `search`, `app`), summary caching, and reuse of summaries from version 1.
- A daily cap on calls made with the server's own API key, for public demos.
- Automated tests and a GitHub workflow that runs them.
- Documentation: what changed and why, and how to put the demo online on Hugging Face Spaces.

## 1.0 (tag `v1.0-bootcamp`)

The notebook built as the capstone of the Applied AI Bootcamp by allWomen. See [`legacy/`](legacy/).

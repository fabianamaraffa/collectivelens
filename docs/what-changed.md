# What changed from version 1 to version 2, and why

This page is for anyone who wants to understand the update without reading code. Version 1 is still here, unchanged, in [`legacy/`](../legacy/), and tagged `v1.0-bootcamp` in git. Nothing was deleted.

The short version: version 1 worked as a bootcamp demo, but some parts did less than it claimed, some parts would stop working, and the interface was a notebook. Version 2 fixes those, adds an honest way to check the results, and moves the app out of the notebook.

## 1. What version 1 claimed and what it did

**Before.** The README of version 1 mentioned BART, HDBSCAN, TF-IDF and UMAP. None of them were in the code. The code used a sentence-embedding model, KMeans and KeyBERT.

**Now.** The README describes only what the code does, and it lists what has not been checked.

**Why it matters.** Anyone who reads a project to judge it will compare the README with the code. A README that promises more than the code does costs trust.

## 2. Search that reads the whole document

**Before.** Each document was turned into one block of numbers by a model that only reads about the first 128 tokens (word pieces), roughly the first paragraph. For a 30-page report that means a few sentences of the introduction. Everything else was invisible to search, to topics and to clusters.

**Now.** Each document is cut into passages of about 180 words, with a small overlap so that no sentence is cut in half unseen. Every passage is turned into numbers, and a search compares your question with all passages. Results show the best passages and mark the matched words, so you can see why a document was found.

**Why it matters.** This is the biggest quality gain in the update. A search tool is only as good as the part of the document it can see.

## 3. A better search model, with a safe fallback

**Before.** `paraphrase-multilingual-MiniLM-L12-v2`, a small model made for short sentences.

**Now.** `BAAI/bge-m3`, a multilingual model that reads long passages and is released under the MIT licence. It is a larger download (about 2 GB). If it cannot be loaded (no internet, not enough memory), the app does not crash: it switches to plain keyword matching and shows a notice at the top. Keyword mode only finds documents that share words with your question, so it cannot find an Italian document with an English question.

**What I could not check.** The environment where this update was written could not download the model, so the semantic path is covered by tests with a stand-in model but has not been run with the real one. Run the app once on your machine and look at the status line: it should say "Search by meaning".

## 4. "Confidence" was not confidence

**Before.** The dashboard showed "Avg Confidence 47%". That number was the average similarity between each document and the short description of its category. A similarity says how close two texts are. It does not say how often the category is right.

**Now.** The number is called **similarity**. Documents that sit between two topics are flagged ("close to ..."), because there the choice is nearly a coin toss. And there is a real check: add a `gold_topic` column to your metadata with the topic a person would choose for some documents, and `python -m collectivelens eval` reports how often the automatic topic agrees, per topic, with a list of the disagreements.

**Why it matters.** It stops a similarity score from being read as an accuracy score, and it gives you a way to find out how good the topics really are.

## 5. Questions with answers and sources

**Before.** For the top results, Claude wrote a short "Why is this relevant?" text. Each one was a separate paid call, and the explanation could not be checked against anything.

**Now.** That call is gone. The results already show the matching passages. A new tab, **Ask the library**, takes a question, retrieves the best passages and asks Claude to answer using only those, citing them as [1], [2]. The app removes any citation number that does not point to a real passage, warns when an answer cites nothing, and shows the cited passages beside the answer. If the passages do not contain the answer, Claude is told to say so. The instructions also tell Claude to ignore any instructions that might be hidden inside a document.

**Why it matters.** It is more useful (a real answer), cheaper (one call per question, not one per result) and checkable.

## 6. Code that would have stopped working

- **Retired models.** Version 1 used `claude-3-haiku-20240307` and `claude-3-5-haiku-20241022`, which Anthropic has retired. The default is now `claude-haiku-5-5`, and you can change it with one environment variable.
- **New answer format.** Newer models can return a "thinking" block before the text. Version 1 read the first block of every answer, so it would break. Version 2 reads only the text blocks, and turns thinking off for these short tasks.
- **Repeatable language detection.** The language library guessed differently from run to run on the same text. A fixed seed makes it repeatable.
- **One bad file no longer stops everything.** A damaged PDF is skipped with a message.
- **Longer summaries.** Version 1 summarised the first 8,000 characters of a document. Version 2 uses up to 30,000.
- **Summaries are saved.** A summary is paid for once. Summaries you already have from version 1 can be reused (`--summaries-file`).

## 7. From a notebook to an app

**Before.** The app lived inside a notebook, the settings were edited in code, and it launched with a public link (`share=True`).

**Now.** The code is a small Python package with one job per file, a command line (`build`, `eval`, `search`, `app`) and settings in environment variables and a categories file. The app is built with Gradio 6, the current version, and has five tabs: Search, Ask the library, Browse, Insights and About. Filters and the optional API key sit in sections that open and close, so it works on a phone. Dark mode works. Charts use colours that people with colour blindness can tell apart, and the library map uses different marker shapes as well. The public link is off unless you ask for it.

## 8. Groups with readable labels

**Before.** KMeans with eight groups, a number chosen by hand, labelled by KeyBERT.

**Now.** The number of groups is chosen automatically (the value with the best silhouette score, which measures how well separated the groups are). Each group is described by the words that set its documents apart from the other groups. KeyBERT was dropped, so there is one dependency less. The topic categories themselves are unchanged: the same eight, now in a file you can edit.

## 9. A demo that needs no private documents

The original 93 documents belong to the member organisations that wrote them, so they are not in the repository. Version 2 includes **13 fictional documents** in English, Italian and French, written only to show how the app works, with summaries already written. Anyone can run the app without a key and without data, and the same files can run as a free online demo (see [`deploy-hugging-face.md`](deploy-hugging-face.md)).

The topics of these 13 documents agree with the "reference" topics 12 times out of 13 in keyword mode. Do not read this as accuracy: I wrote the documents to fit the topics. The one disagreement is flagged by the app as a close call, which is the behaviour I wanted to show.

## 10. Safety and cost

- **Public demo and your key.** If the demo runs with your own API key, anyone could use it. A daily limit (40 calls by default) stops that. After the limit, visitors still get search and passages, and can paste their own key.
- **Untrusted text.** Documents are treated as data. Anything displayed from a document or from a user is escaped before it reaches the page.
- **No secrets in the repository.** The old notebook had an AWS account number and bucket name in its outputs and read the key from a pasted cell. The cleaned notebook in `legacy/` removes them, and `.gitignore` keeps your library and `.env` out of git.

## What stayed the same

The idea, the eight policy areas, the use of Claude for summaries, the multilingual focus, and the notebook itself (cleaned of secrets, in `legacy/`).

## What has not been checked

Be clear about these before relying on the tool:

1. The real semantic model has not been run end to end (see section 3).
2. Live calls to Claude have not been run. Summaries and answers are covered by tests with a stand-in client that mimics the real response format, including a thinking block and a model that rejects the thinking setting. Run `python -m collectivelens build` and one question with your key to confirm.
3. The topic categories and summaries have not been compared with a hand-labelled sample of your real documents. That is what `eval` is for.
4. The notebook in `legacy/` was cleaned of secrets and updated to a current model, but was not re-run on the original library. Run it once before you publish it.

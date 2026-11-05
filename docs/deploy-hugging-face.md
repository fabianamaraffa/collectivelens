# Put the demo online for free (Hugging Face Spaces)

A Hugging Face Space runs the app on a free shared computer and gives it a public link, so people can try CollectiveLens without installing anything. The free "CPU Basic" machine (2 vCPU, 16 GB of memory) is enough for the demo library. A free Space goes to sleep when nobody uses it and wakes up on the next visit, which takes a little while.

**Public demos use the fictional demo library only.** The documents of your own library belong to their authors, and a public Space shows their full text. Keep your own library on your computer.

## Steps

1. Create a free account on huggingface.co, then **New Space**: choose the **Gradio** SDK, hardware **CPU basic**, visibility **Public**.
2. In this repository, prepare the files the Space needs:

   ```bash
   python scripts/make_space_bundle.py
   ```

   This creates a folder `space-bundle/` with the app, the demo library, the requirements and a `README.md` that starts with the settings block Hugging Face needs (taken from [`huggingface-space-README.md`](huggingface-space-README.md)). The GitHub README is not reused because it has no such block.
3. Put those files in the Space. Either drag them into the Space's **Files** page, or use git:

   ```bash
   git clone https://huggingface.co/spaces/YOUR-NAME/YOUR-SPACE
   cp -r space-bundle/* YOUR-SPACE/
   cd YOUR-SPACE && git add . && git commit -m "CollectiveLens demo" && git push
   ```

   When git asks for a password, use a Hugging Face access token with write permission.
4. Wait for the build. The first one is slow: it installs PyTorch and the search model library, and the line `preload_from_hub: BAAI/bge-m3` in the settings block downloads the search model (about 2 GB) while building, so that restarts are quick. `startup_duration_timeout: 1h` gives the Space time to start.
5. Open the Space and check the status line at the top of the app. It should say **Search by meaning**. If it says **Keyword search** with a notice, the model did not load: open the **Logs** tab of the Space for the reason (the usual ones are not enough memory or a failed download).
6. Add the Space link to the README of the GitHub repository (there is a comment where it goes).

## The API key: read this before adding one

Written answers need an Anthropic API key. You have two ways to offer them:

- **Visitors bring their own key.** Do nothing. They paste it under "Written answers", it is used for their session only and is not saved. Without a key the demo still searches and shows the matching passages. This costs you nothing.
- **You pay for answers.** Add your key in the Space under **Settings, Variables and secrets, New secret**, named `ANTHROPIC_API_KEY`. Anyone who visits could then spend your money. CollectiveLens limits the server key to **40 calls a day** (change it with a variable called `COLLECTIVELENS_DAILY_LIMIT`; `0` turns the server key off). Also set a monthly spend limit in your Anthropic console, because a daily cap resets every day and the counter restarts when the Space restarts.

Never put a key in the code, in the README or in a file in the Space.

## Good to know

- **Restarts forget things.** Persistent storage is no longer offered for Spaces, so files created while the app runs (such as saved search vectors) disappear when the Space restarts. For the demo library that only means a few seconds of recomputing at start.
- **Updating.** After a change in the repository, run the bundle script again and push the result to the Space.
- **A lighter model.** If the free machine struggles, a smaller multilingual model can be set with the variables `COLLECTIVELENS_EMBEDDING_MODEL=intfloat/multilingual-e5-small`, `COLLECTIVELENS_QUERY_PREFIX="query: "` and `COLLECTIVELENS_PASSAGE_PREFIX="passage: "`. This combination has not been tested; check the status line and the quality of the results before keeping it.
- **Privacy notice.** Visitors who paste a key send their question and the matching passages to Anthropic. The About tab says so.

Hugging Face may change its settings and limits. The [Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference) is the place to check.

---
title: CollectiveLens
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Search and ask a library of policy documents (fictional demo data)
startup_duration_timeout: 1h
preload_from_hub:
  - BAAI/bge-m3
---

# CollectiveLens (demo)

Search, ask and explore a library of NGO policy documents in English, Italian and French.

This demo runs on **13 fictional documents** written only to show how the app works. Written answers need an Anthropic API key: paste yours under "Written answers" (it is used for your session only and is not saved). Without a key you still get search, topics, groups, charts and the best matching passages.

Source code and documentation: https://github.com/fabianamaraffa/CollectiveLens

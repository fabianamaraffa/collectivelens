"""Check the automatic topics against topics a person assigned (`gold_topic` in the metadata).

Version 1 reported an "Avg Confidence 47%". That number was the average cosine similarity between
documents and category descriptions, which says nothing about being right. This module measures
agreement with a human instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .library import Library


@dataclass
class TopicReport:
    n: int
    accuracy: float
    top2_accuracy: float
    per_topic: pd.DataFrame
    confusion: pd.DataFrame
    wrong: pd.DataFrame
    margin_right: float
    margin_wrong: float

    def to_markdown(self) -> str:
        lines = [
            f"Documents with a reference topic: **{self.n}**",
            f"Top-1 agreement: **{self.accuracy:.0%}**   Top-2 agreement: **{self.top2_accuracy:.0%}**",
        ]
        if not pd.isna(self.margin_wrong):
            lines.append(f"Average margin to the second topic: {self.margin_right:.3f} when right, "
                         f"{self.margin_wrong:.3f} when wrong")
        lines += ["", self.per_topic.round(2).to_string(), ""]
        if len(self.wrong):
            lines += ["Disagreements:", self.wrong.to_string(index=False)]
        return "\n".join(lines)


def evaluate_topics(lib: Library) -> TopicReport | None:
    """Compare automatic topics with `gold_topic`. Returns None when no document has one."""
    rows = []
    for doc, topic in zip(lib.docs, lib.topics):
        if doc.gold_topic:
            rows.append({
                "title": doc.title, "gold": doc.gold_topic, "predicted": topic.name,
                "second": topic.second, "margin": topic.margin,
                "right": topic.name == doc.gold_topic, "right_top2": doc.gold_topic in (topic.name, topic.second),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    names = lib.topic_names
    per = []
    for name in names:
        tp = int(((df.gold == name) & (df.predicted == name)).sum())
        fp = int(((df.gold != name) & (df.predicted == name)).sum())
        fn = int(((df.gold == name) & (df.predicted != name)).sum())
        per.append({"topic": name, "reference": tp + fn, "predicted": tp + fp,
                    "precision": tp / (tp + fp) if tp + fp else float("nan"),
                    "recall": tp / (tp + fn) if tp + fn else float("nan")})
    confusion = pd.crosstab(df.gold, df.predicted).reindex(index=names, columns=names, fill_value=0)
    wrong = df[~df.right][["title", "gold", "predicted", "second", "margin"]].round({"margin": 3})
    return TopicReport(
        n=len(df), accuracy=float(df.right.mean()), top2_accuracy=float(df.right_top2.mean()),
        per_topic=pd.DataFrame(per).set_index("topic"), confusion=confusion, wrong=wrong,
        margin_right=float(df[df.right].margin.mean()) if df.right.any() else float("nan"),
        margin_wrong=float(df[~df.right].margin.mean()) if (~df.right).any() else float("nan"),
    )

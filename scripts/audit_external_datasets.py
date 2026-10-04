"""Reproduce the annotation-quality numbers in docs/experiments/2026-10-04-dataset-survey.md.

Downloads the two public MI corpora whose raw annotator columns allow an
agreement check (AnnoMI-full, Welivita MI) into a scratch dir and prints:
  - AnnoMI: Fleiss kappa on the 7 transcripts coded by all 10 experts,
    per-annotator kappa vs leave-one-out majority, and per-annotator label
    rates on the single-annotator remainder (annotator effect).
  - Welivita: Cohen kappa between the two crowd annotators (ann1 vs ann2).

Usage: python scripts/audit_external_datasets.py [scratch_dir]
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

URLS = {
    "annomi_full.csv": "https://raw.githubusercontent.com/uccollab/AnnoMI/HEAD/AnnoMI-full.csv",
    "welivita_mi.csv": "https://raw.githubusercontent.com/anuradha1992/Motivational-Interviewing-Dataset/HEAD/MI%20Dataset.csv",
}


def fleiss(df, item, label):
    t = pd.crosstab(df[item], df[label])
    t = t[t.sum(1) == t.sum(1).max()].values.astype(float)
    n, N = t.sum(1)[0], len(t)
    p = t.sum(0) / (N * n)
    P = ((t ** 2).sum(1) - n) / (n * (n - 1))
    return (P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum()), N


def loo_kappas(df, label):
    out = []
    for a in sorted(df.annotator_id.unique()):
        me = df[df.annotator_id == a].set_index("item")[label]
        maj = df[df.annotator_id != a].groupby("item")[label].agg(lambda s: s.value_counts().index[0])
        out.append(round(cohen_kappa_score(me, maj.loc[me.index]), 2))
    return out


def main(scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if not (scratch / name).exists():
            urllib.request.urlretrieve(url, scratch / name)

    f = pd.read_csv(scratch / "annomi_full.csv")
    key = ["transcript_id", "utterance_id"]
    multi = f.groupby(key).filter(lambda x: x.annotator_id.nunique() > 1).copy()
    multi["item"] = multi.transcript_id.astype(str) + "_" + multi.utterance_id.astype(str)
    th, cl = multi[multi.interlocutor == "therapist"].copy(), multi[multi.interlocutor == "client"].copy()
    print(f"AnnoMI 10-rater subset: {multi.transcript_id.nunique()} transcripts, "
          f"{th.item.nunique()} therapist / {cl.item.nunique()} client utterances")
    print("  Fleiss main_therapist_behaviour: %.3f (n=%d)" % fleiss(th, "item", "main_therapist_behaviour"))
    print("  Fleiss client_talk_type:         %.3f (n=%d)" % fleiss(cl, "item", "client_talk_type"))
    for c in ["reflection_subtype", "question_subtype", "therapist_input_subtype"]:
        th[c + "_f"] = th[c].fillna("none")
        print(f"  Fleiss {c:24s}  %.3f (n=%d)" % fleiss(th, "item", c + "_f"))
    print("  per-annotator kappa vs LOO majority, therapist:", loo_kappas(th, "main_therapist_behaviour"))
    print("  per-annotator kappa vs LOO majority, client:   ", loo_kappas(cl, "client_talk_type"))
    # Same utterances for every annotator, so these rates are not confounded by session content.
    cx = th.groupby("annotator_id").reflection_subtype.apply(lambda s: (s == "complex").sum() / s.notna().sum())
    ct = cl.groupby("annotator_id").client_talk_type.apply(lambda s: (s == "change").mean())
    print("  same items, complex share of reflections by annotator:", cx.round(2).tolist())
    print("  same items, change-talk share by annotator:          ", ct.round(2).tolist())

    single = f.groupby(key).filter(lambda x: len(x) == 1)
    s_cl, s_th = single[single.interlocutor == "client"], single[single.interlocutor == "therapist"]
    print("AnnoMI single-annotator part, client talk rate by annotator:")
    print(pd.crosstab(s_cl.annotator_id, s_cl.client_talk_type, normalize="index").round(2).to_string())
    print("AnnoMI single-annotator part, reflection subtype by annotator:")
    print(pd.crosstab(s_th.annotator_id, s_th.reflection_subtype, normalize="index").round(2).to_string())

    w = pd.read_csv(scratch / "welivita_mi.csv")
    lis = w[w.author == "listener"].dropna(subset=["ann1", "ann2"])
    print(f"Welivita listener utterances: {len(lis)}, raw agreement {np.mean(lis.ann1 == lis.ann2):.3f}, "
          f"Cohen kappa {cohen_kappa_score(lis.ann1, lis.ann2):.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mi_dataset_audit"))

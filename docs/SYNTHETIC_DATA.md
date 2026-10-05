# Synthetic data (our generated MISC sets)

Kept apart from [DATASETS.md](DATASETS.md) on purpose. These sets are **ours, regenerable,
and change with every synthesis campaign**, while DATASETS.md catalogues the stable
external/human-labelled corpora. The figures below describe the files on disk as of
2026-10-05; rerun [`notebooks/synthetic_eda.ipynb`](../notebooks/synthetic_eda.ipynb) after
any new generation. How the data is generated: [`src/synth/README.md`](../src/synth/README.md).
Results of training on it: [EXPANSION_RESULTS.md](EXPANSION_RESULTS.md).

Role: **training augmentation only.** Never use these sets as gold or for evaluation.

## Names

Every `data/synth/aug/*.csv` file is **the 1,925 HLQC train rows + the synthetic rows**.
Synthetic rows have `conv_id` starting `synth:`. `data/synth/raw/` holds the pre-verify versions.
The registry loader (`eda.registry.load_synth`) returns the synthetic rows only.

| ID | File | What it is |
|---|---|---|
| `synth.v1` | `Qwen2_5-32B-Instruct-AWQ.csv` | v1 prompt (`prompts_v1`), one code-conditioned 2–4 turn snippet per sample |
| `synth.v2_proto` | `…_proto.csv` | v2 ontology dialogue windows (`prompts_v2`), prototype focus |
| `synth.v2_boundary` | `…_boundary.csv` | v2 windows, confusable-pair focus |
| `synth.v3_hlqcmix` | `…_proto_train.csv` | v3 (`generate.py`), topic mix measured on HLQC. "train" in the name is the mix, not a split |
| `synth.v3_mivmix` | `…_proto_eval.csv` | v3, topic mix measured on the **MIV6.3A test set**. Despite "eval" in the name, this is **training** data |

## Size and labels

| ID | Windows | Units | Labelled units (rest = context) | Counsellor / client codes covered | RCP | TS− | AC+ |
|---|---:|---:|---:|---|---:|---:|---:|
| `synth.v1` | 862 | 2,631 | 862 | 18 / 16 | 8 | 20 | 20 |
| `synth.v2_proto` | 50 | 716 | 352 | 12 / 14 | 0 | 0 | 20 |
| `synth.v2_boundary` | 48 | 645 | 292 | 12 / 14 | 0 | 0 | 17 |
| `synth.v3_hlqcmix` | 50 | 848 | 409 | 14 / 15 | 0 | 0 | 18 |
| `synth.v3_mivmix` | 50 | 788 | 387 | 16 / 16 | 0 | 5 | 14 |

Labels are generator-assigned and checked by an independent verifier; they are not human gold.
Synthetic sets are the only source of the two MISC 2.5 classes with no real example
(RCP, TS−; see DATASETS.md §3), so any result on those classes reflects synthetic supervision only.
Like our gold sets, they use AutoMISC's AC± extension.

## Diversity (`synth.metrics`, vs HLQC gold text)

| ID | distinct-1 | distinct-2 | Self-BLEU (lower = more diverse) | near-duplicate % |
|---|---:|---:|---:|---:|
| `synth.v1` | 0.032 | 0.193 | 0.263 | 30.7 |
| `synth.v2_proto` | 0.128 | 0.549 | 0.047 | 8.8 |
| `synth.v2_boundary` | 0.129 | 0.543 | 0.046 | 7.8 |
| `synth.v3_hlqcmix` | 0.104 | 0.485 | 0.037 | 6.0 |
| `synth.v3_mivmix` | 0.105 | 0.464 | 0.066 | 8.8 |

## Overlap with real data and with each other

Same detector as DATASETS.md §5.1: 5-gram containment, with all datasets indexed together.

| Relation | Detail | Consequence |
|---|---|---|
| contains HLQC train | every aug file includes the 1,925 HLQC gold rows; synthesis exemplars are HLQC gold (`generate.load_real_exemplars`) | never count the aug files' real rows as extra data |
| copies HLQC-train phrases | v2/v3 windows reuse exemplar phrases verbatim. Max containment: v3_hlqcmix 0.20 (low_033), v3_mivmix 0.14 (high_099), v2_proto 0.13 (high_121) | training-side only. Through the HLQC-train copies this touches AnnoMI 44 (0.15) and CASAA Emmy (0.12), which DATASETS.md already excludes from those tests |
| uses test-set statistics | `synth.v3_mivmix` topic proportions were measured on the MIV6.3A test set (`synth.ontology.MIXES["eval"]`); no test text is used | a documented ablation, not a leak of labels or text. Report it as such |
| no test text | `synth.verify` drops windows resembling the eval set; no synthetic window overlaps MIV6.3A gold | — |
| v3 mixes share windows | `synth:proto:train:19` = `synth:proto:eval:19` (0.91), same for window 33 | the two v3 arms are not fully independent |
| v1 internal duplicates | 15 pairs ≥ 0.1, 3 redundant windows (mode collapse) | listed for dropping |

## Split manifest

`data/splits/synthetic.json`, built by `python -m eda.splits`, says these sets are train only.
It lists the 3 redundant v1 windows and the 2 windows shared by the v3 mixes. The real-data
manifests are computed from real data alone, so regenerating synthetic data never changes them.

## Reproduce

```bash
PYTHONPATH=src python -m eda.splits
jupyter nbconvert --to notebook --execute --inplace notebooks/synthetic_eda.ipynb
```

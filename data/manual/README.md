# Human evaluation labels (tracked in git)

These CSVs are **required** for fine-tuning and evaluation. They are the only
`data/` CSVs committed to the repo (all other `*.csv` files remain gitignored).

| File | Utterances | Used by |
|---|---:|---|
| `MIV6.3A_manual.csv` | 821 | `automisc_ft` (5-fold CV), `sft` train split |
| `HLQC_balanced_manual.csv` | 1924 | `sft` LODO held-out test |

If `data/manual/` is empty after `git clone`, either:

```bash
git pull   # after manual CSVs were pushed to the remote
```

or copy from your Mac:

```bash
bash scripts/sync_data_to_mlerp.sh
```

Verify:

```bash
bash scripts/check_data.sh
```

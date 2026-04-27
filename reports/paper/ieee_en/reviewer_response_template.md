# Reviewer Response Template

> Use this skeleton when responding to peer-review reports. Save one file per reviewer (`reviewer1_response.md`, etc.) so revisions stay traceable.

---

## Reviewer N — Summary

**Decision recommended**: [Accept / Minor Revision / Major Revision / Reject]
**Reviewer's overall assessment** (paraphrase in 2–3 sentences):

> *"…the paper benchmarks 27 models on Vietnamese gold and reports a feature-family ablation. The conformal-prediction analysis is interesting but the discussion of long-horizon coverage is thin…"*

We thank Reviewer N for the careful read and constructive feedback. Below we address each point with specific manuscript changes; **page/line references refer to the revised manuscript**.

---

## Point-by-point responses

### R-N.1 — *(Reviewer comment, verbatim or paraphrased)*

> *"… <quote the reviewer> …"*

**Our response.** *(One paragraph: thank the reviewer, clarify what changed, point to the change.)*

**Manuscript update.** Section 4.5, page 5, lines 12–21: we added `<change>`. Figure 3 was re-rendered at 300 DPI with the new error bars. The corresponding CSV `reports/ablation/ablation_summary.csv` is included in the supplementary archive.

---

### R-N.2 — *(next point)*

> *"…"*

**Our response.** …

**Manuscript update.** …

---

### R-N.3 — *(typo / formatting)*

> *"line 142, 'inflate' should be 'inflated'"*

**Our response.** Corrected. Thank you.

**Manuscript update.** Section 5, page 7, line 142.

---

## Summary of changes (across all reviewers)

| § | Change | Driven by |
|---|---|---|
| Abstract | Added explicit note on sentiment-window limitation | R1.1 |
| Sec. 4.5 | Expanded ablation discussion, added Table II rows for $h=20$ Ridge | R2.3 |
| Sec. 4.6 | Added per-fold figure for ElasticNet $h=1$ | R1.4, R3.2 |
| Sec. 5 | Re-wrote *Limitations* paragraph for clarity | R2.5 |
| Refs | Added Angelopoulos 2024, Ekambaram 2024, Rasul 2024, Nguyen 2020 PhoBERT, He 2023 mDeBERTa | R3.1 |

## Files modified

- `reports/paper/ieee_en/main.tex`
- `reports/paper/ieee_en/bib.bib`
- `reports/paper/ieee_en/figures/fig*.png` (300 DPI re-render)
- *(any code changes if reviewers required re-running an experiment)*

## Reproducibility

All revisions are committed to branch `paper-revision-v2` of the public repository (`https://github.com/twangnhat-05/NGHIENCUUKHOAHOC`) and tagged `paper-r1-rebuttal`.

---

*Thank you again for the time and care invested in reviewing our manuscript.*

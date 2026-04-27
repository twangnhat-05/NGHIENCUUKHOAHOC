# Submission Checklist — IEEE Conference

> Tick every box before clicking *Submit*. Use one copy per target venue.

**Target venue**: ____________________________________
**Deadline**: ____________________________________
**Manuscript ID**: ____________________________________

---

## A. Manuscript

- [ ] Title is concise (≤120 chars) and search-friendly.
- [ ] Abstract ≤250 words, no abbreviations on first use.
- [ ] Page count within venue limit (RIVF / SoICT: usually 6 pages; ICONIP / KSE: 8–12).
- [ ] All figures rendered at ≥300 DPI from `scripts/generate_paper_figures.py`.
- [ ] Every figure and table is referenced from the body text.
- [ ] Cross-references compile cleanly (no `??`).
- [ ] All citations in `bib.bib` are cited at least once in `main.tex`; no unused entries.
- [ ] Author names, affiliations, and emails match the venue's metadata page.
- [ ] Acknowledgments section credits collaborators, funding, and tooling assistants honestly.

## B. Reproducibility artefacts

- [ ] Public repository URL inserted in the *Contributions* and *Acknowledgments* sections: <https://github.com/twangnhat-05/NGHIENCUUKHOAHOC>.
- [ ] Repository tagged `paper-submission-vN` and the tag is pushed to GitHub.
- [ ] `README.md` includes a *Reproduce paper numbers* section pointing to `scripts/run_ablation_features.py` and `scripts/generate_paper_figures.py`.
- [ ] `requirements.txt` is pinned and a fresh clone reaches `pytest tests/ -q` passing in <10 min.
- [ ] All artefact CSVs (`reports/ablation/*.csv`, `reports/leaderboard/*.csv`, `reports/figures/*.csv`) are committed and not git-ignored.
- [ ] `LICENSE` is MIT (or a permissive licence the venue accepts) and `CITATION.cff` is up to date.

## C. PDF assembly (Overleaf or local)

- [ ] Project compiles with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` without errors.
- [ ] Final PDF metadata has the correct title, author, and keywords.
- [ ] All hyperlinks open correctly (Overleaf: *Compile* → click the GitHub URL in the rendered PDF).
- [ ] No "draft" or "TODO" markers remain anywhere in the document.

## D. Ethics, integrity, AI disclosure

- [ ] No plagiarised passages — paraphrase, cite, or quote.
- [ ] No data collected from proprietary or paid sources without licence; all sources here are free-tier.
- [ ] AI-tooling disclosure (if any) in *Acknowledgments* matches what was actually used.
- [ ] Human author owns final responsibility for every claim in the paper.
- [ ] No PII or sensitive trading positions anywhere in the data.

## E. Submission system steps (typical EasyChair flow)

- [ ] Logged in to the conference EasyChair / OpenReview portal.
- [ ] Selected correct track (e.g., *AI in Finance*, *Time Series*, *Applications*).
- [ ] Uploaded `main.pdf`.
- [ ] Uploaded supplementary archive: `bib.bib`, `figures/*.png`, `cover_letter.pdf`, `reports/ablation/ablation_summary.csv`.
- [ ] Filled in all required metadata (abstract, keywords, suggested topics).
- [ ] Suggested reviewers entered (if the venue allows it).
- [ ] Conflicts of interest declared honestly.
- [ ] Final *Submit* button clicked **before** the deadline plus a 24h safety margin.
- [ ] Confirmation email saved in a dedicated `submissions/<venue>/` folder.

## F. Post-submission

- [ ] Project tag `paper-submission-vN` is pushed to GitHub.
- [ ] Slack / email reminder set 1 week before notification deadline.
- [ ] Calendar reminder for camera-ready deadline.
- [ ] `MEMORY.md` updated with submission status (so the next next session has context).

---

*Last reviewed:* ____________________________________

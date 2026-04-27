# Cover Letter — IEEE Conference Submission

> Template for RIVF / SoICT / KSE / ICONIP. Customise the highlighted parts before submitting.

---

**To**: The Program Chairs, **[CONFERENCE NAME]** [YEAR]
**Subject**: Manuscript submission — *Foundation Models versus Engineered-Feature Linear Regression for Vietnamese Gold Price Forecasting under Regime Shifts*

Dear Program Chairs,

Please find attached our manuscript for consideration as a regular paper at **[CONFERENCE NAME]** [YEAR]. We confirm that the paper is original work, has not been published elsewhere, and is not under simultaneous review at any other venue.

## Why this paper fits your venue

[CONFERENCE NAME] focuses on **[machine learning / data science / Vietnamese applications / emerging-market analytics — pick the most relevant]**. Our work contributes to this scope in three ways:

1. **First systematic benchmark** of 27 forecasting models — including modern foundation models (Chronos-Bolt, TimesFM, Lag-Llama wrappers) — on the Vietnamese SJC physical-gold market, an emerging-market asset that has received limited attention in the international literature.
2. **A faithfully-reported feature-family ablation** that quantifies the marginal contribution of lag, returns, technical, macro, calendar, and (mDeBERTa zero-shot) sentiment features. We honestly report sentiment as a no-op in the current evaluation window because the news-archive coverage does not overlap our cross-validation folds — illustrating a methodological pitfall worth documenting for the community.
3. **Per-fold conformal coverage analysis** under the 2024 SJC rally regime shift, showing where vanilla split conformal collapses (9% empirical coverage) and where Adaptive Conformal Inference (ACI) maintains target coverage. This complements the growing body of work on conformal prediction for time-series under distribution shift.

The work was conducted under the **TDTU Sinh Viên NCKH 2025–2026** programme. All code, data, leaderboards, ablation tables, conformal results, a Streamlit / PWA dashboard, FastAPI service, Telegram bot, weekly retraining pipeline, and Docker images are open-sourced under the MIT licence at <https://github.com/twangnhat-05/NGHIENCUUKHOAHOC>. We believe the artefact is reproducible by reviewers in under ten minutes on free-tier infrastructure.

## Suggested reviewers (optional)

Please consider the following reviewers with relevant expertise:

* **[Reviewer 1]** — [affiliation], expertise in [time-series forecasting / foundation models / Vietnamese macro] — [email].
* **[Reviewer 2]** — [affiliation], expertise in conformal prediction or financial time-series — [email].

We have no conflict of interest with any potential reviewer beyond standard co-author / advisor relationships.

## Author contributions and contact

WangNhat (corresponding author) led the data engineering, model implementation, experiment design, and manuscript drafting. Statistical-test design and architecture review were conducted all final decisions, scientific claims, and reported numbers are the human author's responsibility.

We thank you in advance for considering our submission and look forward to your feedback.

Sincerely,

**WangNhat**
Faculty of Information Technology, Ton Duc Thang University
Email: dev2@wolffungame.com
GitHub: <https://github.com/twangnhat-05>
ORCID: *(add when ready)*

---

*Submitted on:* **[DATE]**
*Manuscript ID:* **[to be assigned]**

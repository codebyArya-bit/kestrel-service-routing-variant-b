# Recording guide (2 minutes 50 seconds)

Record the actual tool with synthetic inputs and public source; keep private metrics, customer data and artifacts off screen. This guide is not a completed video.

| Time | Show | Narration |
|---|---|---|
| 0:00–0:20 | Actual retained prompt/task brief if available; `docs/prompt-version-log.md` | Codex assisted development; prediction uses local scikit-learn. Missing historical prompts are not reconstructed. |
| 0:20–0:50 | Candidate grid and baseline code in `train.py` | **Tried:** word-only, word plus character, metadata alternatives and a bot-imitation baseline trained on old bot labels. These are recorded experiments; show no private scores and claim no new run. |
| 0:50–1:25 | Genuine source versions/git diffs; `train.py`, `routing/model.py`, `serve.py` and nested-JSON test | **Changed:** candidate target uses joined closing-team outcomes as an imperfect proxy instead of old bot labels as ground truth. Explain chosen-versus-runner-up contribution reasons, vague-input review guard and nested JSON handling. Show evidenced changes without inventing historical versions. |
| 1:25–2:00 | `python serve.py`; authored installation request and vague callback request | Show suggested team, evidence, alternatives and review status. Synthetic behavior does not establish operational accuracy. |
| 2:00–2:30 | Candidate selection code and generic method/pilot docs | **Threw away:** metadata candidates were evaluated and discarded after development selection. Payment-mention-only policy routing, old bot labels as ground truth, unconditional bot retirement and paid inference calls were rejected proposals, not claims of implemented experiments. Final scope uses policy-aware rules, proxy caveats and a supervised pilot. |
| 2:30–2:50 | Generic validation/cost docs and genuine verified links if available | Selection was frozen before holdout; paid prediction API calls are zero, hosting/review costs unmeasured. Stop before three minutes; supply actual duration, real recording URL and candidate-confirmed hours separately. |

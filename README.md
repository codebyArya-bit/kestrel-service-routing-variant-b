# Kestrel service routing prototype

A local employee screen and JSON endpoint suggest one of seven service teams and show review status, alternatives and model evidence. Public startup trains only the authored synthetic examples in `demo/synthetic.csv`; it needs no API key or private files. Synthetic behavior is illustrative, not a production validation result.

## Run

Python 3.11 or newer is recommended (Python 3.12 is a suitable choice). From this directory:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe serve.py
```

If `py` is unavailable, install Python from python.org with **Add python.exe to PATH**, reopen PowerShell, and use `python -m venv .venv` instead of the first command. Windows uses `Scripts\python.exe`; Linux/macOS use `bin/python`. No activation command is required.

For the separately supplied trusted restricted bundle, after verifying its hashes:

```powershell
.\.venv\Scripts\python.exe serve.py --model final/model.pkl
```

Run tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

### Linux / macOS

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python serve.py
```

Open http://127.0.0.1:8766. POST a JSON object to `/api/route`:

```sh
curl -s http://127.0.0.1:8766/api/route -H 'Content-Type: application/json' -d '{"request_text":"Please install my new purifier","channel":"web","product_family":"appliance","warranty_status":"unknown"}'
```

`request_text` is required. Optional enum values are listed on the screen. Scores are uncalibrated model support, not probabilities of correctness. Reasons describe model evidence; they are not causal explanations. Vague inputs can require review; multi-intent requests can still pass a keyword guard.

## Verify and private operation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

An authorized operator can run `python serve.py --model /trusted/path/model.pkl` to load a separately supplied private artifact. Pickle can execute code: only load trusted, verified artifacts. The service binds loopback and suppresses intake request logging. Production authentication, TLS, hosting and queue integration need separate work.

The restricted workflow is `python train.py --data /authorized/input/pack --output /new/restricted/output`; output must be fresh because frozen artifacts are protected against overwrite. It creates the model, ordered CSV and evaluation/freeze manifests together. Do not upload private inputs, fitted artifacts or operational evidence to a public repository. Existing frozen artifact hashes must be checked rather than silently regenerated. Serving display aliases and reason wording can change independently of frozen score decisions.

See `docs/` for generic method, limitations, costs and handoff templates. Operational evidence and completed submission answers are distributed separately. No recording or published repository URL is claimed here.

## Handoff status

Senior whole-branch review completed with no critical or important defects. Final scoped fix review passed; candidate submission remains pending. Actual Task 2 hours and an under-three-minute recording URL remain unconfirmed. Public repository and the designated public Drive folder were verified on 8 October 2026.

Public code: https://github.com/codebyArya-bit/kestrel-service-routing-variant-b . Publish only public-safe code and synthetic examples. Restricted artifacts belong only to the owner or authorized engagement recipients. Complete the unresolved form fields from observed facts before submitting.

An authorized local operator can package existing frozen artifacts with `python scripts/package.py --output /absolute/output/path --expected-prediction-sha256 VERIFIED_SHA256`. This command requires an authorized source checkout containing the omitted private input pack and restricted handoff files. Downloaded bundles intentionally omit those inputs and cannot recreate packaging alone. It validates frozen model/freeze hashes against evaluation evidence and existing predictions without training or regenerating them.

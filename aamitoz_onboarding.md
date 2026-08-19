# Aamitoz's Onboarding Guide (Windows)

Welcome to the Karcytics-cytometrics project! Since we enforce a strict cryptographic trust model to prevent tampered plugins, you'll need to set up your cryptographic identity and get authorized by Kalaimaran before you can run or commit code.

## 1. Setup Local Environment

1. Install [Git](https://git-scm.com/), [Python (>=3.11)](https://www.python.org/downloads/), and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Open PowerShell and clone the repositories:
   ```powershell
   # Create a working folder
   mkdir Karcytics-Workspace
   cd Karcytics-Workspace

   # Clone the repositories
   git clone https://github.com/KalaimaranB/Karcytics-cytometrics.git
   git clone https://github.com/KalaimaranB/Karcytics-SDK.git
   ```
3. Set up the environment and install dependencies (including the SDK, via the `../Karcytics-SDK` editable path already declared in `pyproject.toml`):
   ```powershell
   cd Karcytics-cytometrics
   uv sync --all-extras
   ```
4. Install the git hooks:
   ```powershell
   uv run pre-commit install
   uv run pre-commit install --hook-type pre-push
   ```

## 2. Cryptographic Identity Setup

1. Generate your developer keys:
   ```powershell
   karcytics-sdk init-identity
   ```
   This generates a private and public key in your Windows user profile at `%USERPROFILE%\.karcytics\dev_keys`.
   *⚠️ Keep `private.key` secret and never commit it!*

2. Find the file at `%USERPROFILE%\.karcytics\dev_keys\public.pub` and send this file to Kalaimaran.

*(Wait for Kalaimaran to return a `delegation.json` file)*

3. Once Kalaimaran sends you the `delegation.json` file, place it in:
   `%USERPROFILE%\.karcytics\dev_keys\delegation.json`

## 3. Daily Workflow

Whenever you make changes to the plugin, they must be cryptographically signed, linted, type-checked, and pass the unit test suite before committing. This is all automated via `pre-commit` (installed in step 1.4):

Commit normally — `git commit` now runs ruff (lint + format), mypy, a pip-audit dependency scan, the unit test suite, and finally `karcytics-sdk sign`, which updates `security.json`, `signature.bin`, and `trust_chain.json` using your cryptographic identity before completing the commit. Your signatures automatically chain back to Kalaimaran's authority, making them valid! `git push` additionally runs a read-only check that the signed ledger still matches the tree.

---

# Kalaimaran's Admin Steps

## 1. Delegating Trust to Aamitoz
When Aamitoz sends you his `public.pub` file:
1. Save it somewhere on your machine.
2. Run the delegation command:
   ```bash
   karcytics-sdk delegate /path/to/his/public.pub "Aamitoz Sharma"
   ```
3. Send the generated `delegation_aamitoz_sharma.json` file back to Aamitoz (he will rename it to `delegation.json`).

## 2. GitHub Secrets (CI Signing)
CI signing now uses a single project key plus a delegation file committed to the repo (`.ci_keys/runner_delegation.json`), rather than the old two-secret developer-identity setup. Repository secrets:
1. Go to **Settings > Secrets and variables > Actions**.
2. **Delete** any old secrets you aren't using anymore: `BIOPRO_SIGNING_PRIVATE_KEY`, `BIOPRO_SIGNING_DELEGATION`, `BIOPRO_DEV_DELEGATION`, `BIOPRO_DEV_PRIVATE_KEY`, `BIOPRO_PROJECT_PRIVATE_KEY`, `DIST_PAT` (the Distribution repo's `registry.json` no longer needs a per-release PR, so this token isn't needed here anymore).
3. **Add/keep**: `KARCYTICS_PROJECT_PRIVATE_KEY`
   - **Value**: The private key generated for `.ci_keys/public.pub` (this plugin's own dedicated CI runner identity — not your personal `~/.karcytics/dev_keys` key). `.github/workflows/release.yml`'s `project-sign` step reads it via this env var and appends `.ci_keys/runner_delegation.json` to the trust chain.

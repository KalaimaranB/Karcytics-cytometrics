# Aamitoz's Onboarding Guide (Windows)

Welcome to the BioPro-cytometrics project! Since we enforce a strict cryptographic trust model to prevent tampered plugins, you'll need to set up your cryptographic identity and get authorized by Kalaimaran before you can run or commit code.

## 1. Setup Local Environment

1. Install [Git](https://git-scm.com/) and [Python (>=3.11)](https://www.python.org/downloads/).
2. Open PowerShell and clone the repositories:
   ```powershell
   # Create a working folder
   mkdir BioPro-Workspace
   cd BioPro-Workspace

   # Clone the repositories
   git clone https://github.com/KalaimaranB/BioPro-cytometrics.git
   git clone https://github.com/KalaimaranB/BioPro-SDK.git
   ```
3. Set up the Python virtual environment for the plugin:
   ```powershell
   cd BioPro-cytometrics
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies and the SDK:
   ```powershell
   pip install PyQt6 pytest
   pip install -e ..\BioPro-SDK
   ```

## 2. Cryptographic Identity Setup

1. Generate your developer keys:
   ```powershell
   biopro-sdk init-identity
   ```
   This generates a private and public key in your Windows user profile at `%USERPROFILE%\.biopro\dev_keys`.
   *⚠️ Keep `private.key` secret and never commit it!*

2. Find the file at `%USERPROFILE%\.biopro\dev_keys\public.pub` and send this file to Kalaimaran.

*(Wait for Kalaimaran to return a `delegation.json` file)*

3. Once Kalaimaran sends you the `delegation.json` file, place it in:
   `%USERPROFILE%\.biopro\dev_keys\delegation.json`

## 3. Daily Workflow

Whenever you make changes to the plugin, they must be cryptographically signed. We have automated this using a git hook!

1. **Enable the hook (Run this once)**:
   ```powershell
   git config core.hooksPath .githooks
   ```

2. **Commit normally**:
   Now, whenever you run `git commit`, the hook will automatically run `biopro-sdk sign` under the hood using your cryptographic identity, updating `manifest.json`, `security.json`, and the signatures before completing the commit. Your signatures will automatically chain back to Kalaimaran's authority, making them valid!

---

# Kalaimaran's Admin Steps

## 1. Delegating Trust to Aamitoz
When Aamitoz sends you his `public.pub` file:
1. Save it somewhere on your machine.
2. Run the delegation command:
   ```bash
   biopro-sdk delegate /path/to/his/public.pub "Aamitoz Sharma"
   ```
3. Send the generated `delegation_aamitoz_sharma.json` file back to Aamitoz (he will rename it to `delegation.json`).

## 2. GitHub Secrets Updates (Upgraded Build Pipeline)
The old CI pipeline keys have been deprecated in favor of the new V2 security ledger. I have upgraded `.github/workflows/release.yml` in this repository to enforce the new checks.

You must update the GitHub Secrets for this repository:
1. Go to **Settings > Secrets and variables > Actions**.
2. **Delete** any old secrets you aren't using anymore: `BIOPRO_DEV_DELEGATION`, `BIOPRO_DEV_PRIVATE_KEY`, `BIOPRO_PROJECT_PRIVATE_KEY`.
3. **Add New Secret**: `BIOPRO_SIGNING_PRIVATE_KEY`
   - **Value**: The full contents of your `~/.biopro/dev_keys/private.key` (or `~/.biopro/dev_private_key.pem`).
4. **Add New Secret**: `BIOPRO_SIGNING_DELEGATION`
   - **Value**: The full contents of your `~/.biopro/dev_keys/delegation.json`.
5. Keep **`DIST_PAT`** as it is still required to automatically update the BioPro-Distribution repository.

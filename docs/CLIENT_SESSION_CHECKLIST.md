# Client Session Checklist — Google Authorization & Go-Live

The exact sequence for the one-time session with David. Everything code-side
is already built, tested, and pushed; this session only connects Google and
verifies the first cloud run. Budget roughly 45–60 minutes.

Ground rules for the whole session:

- **Never ask for, type, or store David's Google password.** All sign-in
  happens on Google's own pages, on a machine where David enters his own
  credentials.
- The production OAuth scope is exactly one, non-sensitive scope:
  `https://www.googleapis.com/auth/drive.file` — access only to files this
  app creates. No Sheets scope is requested; none is needed.
- **Do not create the production Drive folder or spreadsheet by hand in the
  browser.** Files made by hand are invisible under `drive.file`. The setup
  helper creates both through the API — that is what makes them visible to
  the automation. (The helper's `--scope-drive-full` flag exists only as a
  troubleshooting escape hatch; it requests the **Restricted** full-Drive
  scope and is not the production path.)

---

## 1. Google Cloud project (David signed in)

- [ ] David signs into the Google account that should own everything.
- [ ] Create a Google Cloud project — suggested name: **NFL Weekly
      Automation** — under David's account, so he owns it.
- [ ] Enable the **Google Drive API**.
- [ ] Enable the **Google Sheets API**.
- [ ] Configure the OAuth consent screen (External; app name, support email).
- [ ] Create an **OAuth client ID**, application type **Desktop app**;
      download the client-secrets JSON.

## 2. Publish BEFORE authorizing — critical

- [ ] Set the OAuth consent screen's **publishing status to "In
      production"** *now*, before generating any token.

  Why this ordering matters: refresh tokens minted while the app is in
  "Testing" are revoked by Google after **7 days** — the Friday automation
  would die silently one week after handoff. Because the app requests only
  the non-sensitive `drive.file` scope, publishing requires **no Google
  verification review** and shows no "unverified app" warning.

## 3. One-time authorization + resource creation

- [ ] On a machine with a browser and this repository:

  ```powershell
  python scripts/google_auth_setup.py --client-secrets client_secret.json
  ```

- [ ] Confirm the console shows exactly one requested scope:
      `https://www.googleapis.com/auth/drive.file`.
- [ ] David approves on Google's consent page.
- [ ] The helper creates the **NFL Weekly Model** Drive folder (with its
      `Model/` subfolder), creates the **NFL Weekly Analytics** spreadsheet
      with all five tabs, moves the Sheet into the folder, and prints five
      values.

## 4. GitHub secrets

- [ ] At `https://github.com/haris7817/nfl-weekly-automation/settings/secrets/actions`,
      add the five printed values as repository secrets:
      `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN`,
      `GOOGLE_DRIVE_FOLDER_ID`, `GOOGLE_SHEET_ID`.
- [ ] **Delete `client_secret.json` from the machine.** Do not paste any of
      the five values into chat, email, or screenshots.

## 5. First cloud run

- [ ] GitHub → *Actions* → **NFL Weekly Automation** → *Run workflow* (no
      inputs).
- [ ] Watch the logs: correct season/week detected; matchups found; model
      trained (first run) and persisted to the Drive `Model/` folder;
      projections generated; CSVs uploaded; all five Sheet tabs populated;
      exit code 0.

## 6. Idempotency check

- [ ] Trigger *Run workflow* a second time.
- [ ] Confirm: no duplicate Drive folders or files (same files updated in
      place), no duplicate rows in *Prediction History*, one new *Run Log*
      row per run.

## 7. Handoff

- [ ] Show David the Drive folder, the Sheet tabs, and what each means
      (including the fair-spread sign convention: negative = home favored).
- [ ] Show the manual *Run workflow* button and the Friday 6:00 PM Pacific
      schedule.
- [ ] Explain the retraining cadence (every 4 weeks, plus new-season and
      forced retrains) and where model provenance lives (*Model Info* tab).
- [ ] Explain where to look when something fails (Actions logs; CSV
      artifacts on every run, even failed ones).
- [ ] Arrange repository access: invite David as a collaborator, or
      transfer the repo to his GitHub account.
- [ ] Confirm David knows he can revoke the app's access at any time at
      [myaccount.google.com/permissions](https://myaccount.google.com/permissions).
- [ ] Record the timestamp of the final successful run.

---

## If something goes wrong

| Symptom | Likely cause / fix |
|---|---|
| `invalid_grant` on a later run | Token was minted while the app was in "Testing" (7-day revocation). Publish to "In production", rerun the helper, update `GOOGLE_REFRESH_TOKEN`. |
| Google 404 for folder or Sheet | The ID belongs to a hand-made item, invisible under `drive.file`. Use the IDs the helper printed. |
| No refresh token returned | The account already authorized this OAuth client once. Remove the app at [myaccount.google.com/permissions](https://myaccount.google.com/permissions) and rerun the helper. |
| Scheduled run absent | GitHub disables schedules after 60 days of repo inactivity; any commit or manual run re-enables. Scheduled starts can also lag under platform load. |

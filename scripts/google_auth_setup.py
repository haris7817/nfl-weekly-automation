"""
One-time Google authorisation helper.
=====================================

Run this ONCE, with the client present, on a machine that has a browser:

    python scripts/google_auth_setup.py --client-secrets client_secret.json

What it does
------------
1. Opens Google's consent screen so the client signs in and approves the
   requested access. The client types their password into Google's own
   page - never into this tool, and never shares it with anyone.
2. Captures the resulting **refresh token**, which is what lets the
   unattended Friday job get fresh access tokens while the client's PC is
   off.
3. Creates the Drive folder ("NFL Weekly Model") and the spreadsheet
   ("NFL Weekly Analytics") with all five tabs.
4. Prints the five values to paste into GitHub Actions secrets.

Why it creates the Drive folder and Sheet
-----------------------------------------
The automation uses the narrow ``drive.file`` scope, which grants access
only to files the app itself created. That is deliberate - it means this
integration can never read the rest of the client's Drive. The trade-off
is that a folder the client makes by hand in the browser would be
invisible to it, so the helper creates them through the API instead.

If the client insists on an existing hand-made folder, rerun with
``--scope-drive-full`` and pass ``--drive-folder-id``; that requests the
broader ``drive`` scope. Prefer the default.

Security
--------
* The client secret file and the token never leave this machine.
* Nothing here writes into the repository; ``.local/`` is git-ignored.
* Delete the downloaded client-secrets JSON once the GitHub secrets are
  saved. If a secret ever leaks, revoke it in Google Cloud Console and
  rerun this helper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import DEFAULT_GOOGLE_SCOPES  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.net import enable_system_trust_store  # noqa: E402

DRIVE_FULL_SCOPE = "https://www.googleapis.com/auth/drive"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"

DEFAULT_FOLDER_NAME = "NFL Weekly Model"
DEFAULT_SHEET_NAME = "NFL Weekly Analytics"


def authorise(client_secrets: Path, scopes, use_console: bool):
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), list(scopes))

    if use_console:
        # Fallback for a machine with no usable browser: the client opens
        # the URL elsewhere and pastes the resulting code back.
        credentials = flow.run_console()
    else:
        print("\nA browser window will open for Google sign-in.")
        print("Sign in as the account that should OWN the Drive folder and Sheet.\n")
        credentials = flow.run_local_server(
            port=0,
            prompt="consent",
            access_type="offline",
            authorization_prompt_message="Opening {url}",
            success_message=(
                "Authorisation complete. You can close this tab and return "
                "to the terminal."
            ),
        )

    if not credentials.refresh_token:
        raise SystemExit(
            "Google did not return a refresh token. This usually means the "
            "account has already authorised this OAuth client. Remove the "
            "app at https://myaccount.google.com/permissions and rerun."
        )
    return credentials


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time Google authorisation and resource setup."
    )
    parser.add_argument(
        "--client-secrets",
        type=Path,
        default=BASE_DIR / "client_secret.json",
        help="OAuth client JSON downloaded from Google Cloud Console.",
    )
    parser.add_argument("--folder-name", default=DEFAULT_FOLDER_NAME)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument(
        "--drive-folder-id",
        default=None,
        help="Reuse an existing folder instead of creating one.",
    )
    parser.add_argument(
        "--sheet-id",
        default=None,
        help="Reuse an existing spreadsheet instead of creating one.",
    )
    parser.add_argument(
        "--scope-drive-full",
        action="store_true",
        help="Request full Drive access (only needed to write into a folder "
        "the client created by hand in the browser).",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Use the copy/paste console flow instead of a local browser.",
    )
    args = parser.parse_args()

    setup_logging("INFO", to_file=False)
    enable_system_trust_store()

    if not args.client_secrets.is_file():
        print(
            f"\nERROR: {args.client_secrets} not found.\n\n"
            "In Google Cloud Console:\n"
            "  1. create/select a project (suggested name: NFL Weekly Automation)\n"
            "  2. enable the Google Drive API and the Google Sheets API\n"
            "  3. configure the OAuth consent screen\n"
            "  4. create an OAuth client ID of type 'Desktop app'\n"
            "  5. download the JSON and pass it with --client-secrets\n",
            file=sys.stderr,
        )
        return 2

    scopes = (
        [DRIVE_FULL_SCOPE, SHEETS_SCOPE]
        if args.scope_drive_full
        else list(DEFAULT_GOOGLE_SCOPES)
    )
    print("Requesting scopes:")
    for scope in scopes:
        print(f"  - {scope}")

    credentials = authorise(args.client_secrets, scopes, args.console)

    from googleapiclient.discovery import build

    from src.google_drive import create_root_folder, ensure_model_folder
    from src.google_sheets import ALL_TABS, create_spreadsheet, ensure_tabs

    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)

    folder_id = args.drive_folder_id
    if folder_id:
        print(f"\nUsing the existing Drive folder {folder_id}.")
    else:
        folder_id = create_root_folder(drive, args.folder_name)
        print(f"\nCreated Drive folder {args.folder_name!r}.")

    # Create the Model subfolder now so the first cloud run has somewhere
    # to persist the trained model.
    ensure_model_folder(drive, folder_id)
    print("Created the 'Model' subfolder for model persistence.")

    sheet_id = args.sheet_id
    if sheet_id:
        ensure_tabs(sheets, sheet_id)
        print(f"Using the existing spreadsheet {sheet_id} (tabs verified).")
    else:
        sheet_id = create_spreadsheet(sheets, args.sheet_name)
        print(f"Created spreadsheet {args.sheet_name!r} with tabs: "
              f"{', '.join(ALL_TABS)}.")
        # Move the new Sheet into the client's folder so everything lives
        # together. Harmless if it fails - the Sheet still works.
        try:
            file = drive.files().get(fileId=sheet_id, fields="parents").execute()
            drive.files().update(
                fileId=sheet_id,
                addParents=folder_id,
                removeParents=",".join(file.get("parents", [])),
                fields="id, parents",
            ).execute()
            print("Moved the spreadsheet into the NFL Weekly Model folder.")
        except Exception as exc:
            print(f"(Could not move the Sheet into the folder: {exc})")

    client_config = json.loads(args.client_secrets.read_text())
    installed = client_config.get("installed") or client_config.get("web") or {}

    print("\n" + "=" * 72)
    print("ADD THESE FIVE GITHUB ACTIONS SECRETS")
    print("Repository -> Settings -> Secrets and variables -> Actions -> New secret")
    print("=" * 72)
    print(f"GOOGLE_CLIENT_ID\n    {installed.get('client_id', '')}\n")
    print(f"GOOGLE_CLIENT_SECRET\n    {installed.get('client_secret', '')}\n")
    print(f"GOOGLE_REFRESH_TOKEN\n    {credentials.refresh_token}\n")
    print(f"GOOGLE_DRIVE_FOLDER_ID\n    {folder_id}\n")
    print(f"GOOGLE_SHEET_ID\n    {sheet_id}\n")
    print("=" * 72)
    print(
        "\nSpreadsheet: "
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    )
    print(f"Drive folder: https://drive.google.com/drive/folders/{folder_id}")
    print(
        "\nNEXT STEPS\n"
        "  1. Paste the five values above into GitHub Actions secrets.\n"
        "  2. Delete the client_secret.json file from this machine.\n"
        "  3. Trigger Actions -> NFL Weekly Automation -> Run workflow.\n"
        "\nThese values are secrets: do not paste them into email, chat or "
        "screenshots.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

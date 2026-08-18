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
   off. The flow always requests offline access with forced consent
   (``access_type=offline``, ``prompt=consent``), so Google issues a new
   refresh token on every run - including reruns, with no need to revoke
   the earlier grant.
3. Creates - or, on a rerun, REUSES - the Drive folder ("NFL Weekly
   Model") and the spreadsheet ("NFL Weekly Analytics") with all five
   tabs. Under the drive.file scope the file<->app association is bound
   to the OAuth client, not to a token, so resources created by an
   earlier authorization are rediscovered instead of duplicated.
4. Ends with the five GitHub Actions secret values as the VERY LAST
   output. They are displayed exactly once and are not written anywhere
   else - copy them before closing the window.

If Google does not return a refresh token, the helper FAILS loudly
instead of reporting success: without that token the Friday automation
cannot run, so a "success" without it would be a lie.

Why it creates the Drive folder and Sheet
-----------------------------------------
The automation uses the narrow ``drive.file`` scope, which grants access
only to files the app itself created. That is deliberate - it means this
integration can never read the rest of the client's Drive. The trade-off
is that a folder the client makes by hand in the browser would be
invisible to it, so the helper creates them through the API instead.

If the client insists on an existing hand-made folder, rerun with
``--scope-drive-full`` and pass ``--drive-folder-id``. That requests the
full ``drive`` scope, which Google classifies as **Restricted** (broad
account-wide Drive access, subject to the heaviest verification tier).
It exists strictly as a troubleshooting escape hatch and is NOT the
normal production path - the production workflow uses ``drive.file`` only.

Before generating the FINAL production refresh token, the OAuth consent
screen must be moved out of "Testing" and published ("In production").
Refresh tokens minted while the app is in Testing are revoked by Google
after 7 days, which would silently kill the Friday automation a week
after handoff.

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

#: Restricted scope used only by the --scope-drive-full escape hatch. The
#: full drive scope covers every Sheets method this project calls, so no
#: additional Sheets scope is requested even in that mode.
DRIVE_FULL_SCOPE = "https://www.googleapis.com/auth/drive"

SPREADSHEET_MIME = "application/vnd.google-apps.spreadsheet"

DEFAULT_FOLDER_NAME = "NFL Weekly Model"
DEFAULT_SHEET_NAME = "NFL Weekly Analytics"

SECRET_NAMES = (
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "GOOGLE_REFRESH_TOKEN",
    "GOOGLE_DRIVE_FOLDER_ID",
    "GOOGLE_SHEET_ID",
)


def select_scopes(scope_drive_full: bool = False) -> list:
    """Scopes to request at consent time.

    Normal production path: exactly ``drive.file`` (non-sensitive).
    Escape hatch: exactly the full ``drive`` scope (Restricted) - never
    both, and never the sensitive ``spreadsheets`` scope.
    """
    if scope_drive_full:
        return [DRIVE_FULL_SCOPE]
    return list(DEFAULT_GOOGLE_SCOPES)


def authorise(client_secrets: Path, scopes, use_console: bool):
    """Run the consent flow and return credentials WITH a refresh token.

    ``access_type="offline"`` asks Google for a refresh token and
    ``prompt="consent"`` forces the consent screen even when the account
    has authorised this client before - Google only reissues a refresh
    token on a consented exchange, so both are required for reruns to
    work without revoking anything.

    Raises ``SystemExit`` if no refresh token comes back: that outcome
    must never be reported as success.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), list(scopes))

    flow_kwargs = {
        "port": 0,
        "access_type": "offline",
        "prompt": "consent",
        "authorization_prompt_message": "Opening {url}",
        "success_message": (
            "Authorisation complete. You can close this tab and return "
            "to the terminal."
        ),
    }
    if use_console:
        # run_console() was removed from google-auth-oauthlib in 1.0; the
        # supported fallback is the local server without a browser launch:
        # the URL is printed for the user to open by hand.
        flow_kwargs["open_browser"] = False
        print("\nNo browser will be opened. Copy the URL below into one,")
        print("sign in as the account that should OWN the Drive folder and")
        print("Sheet, and approve access.\n")
    else:
        print("\nA browser window will open for Google sign-in.")
        print("Sign in as the account that should OWN the Drive folder and Sheet.\n")

    credentials = flow.run_local_server(**flow_kwargs)

    if not credentials.refresh_token:
        raise SystemExit(
            "\n"
            "========================================================\n"
            " FAILED: Google did not return a refresh token.\n"
            "\n"
            " Without a refresh token the weekly automation cannot\n"
            " run unattended, so this authorization DID NOT SUCCEED.\n"
            "\n"
            " The authorization needs to be repeated with a forced\n"
            " consent screen. This helper already requests that\n"
            " (access_type=offline, prompt=consent), so simply run it\n"
            " again. If it still happens, remove the app's access at\n"
            " https://myaccount.google.com/permissions and run once\n"
            " more.\n"
            "========================================================"
        )
    return credentials


def provision(
    drive,
    sheets,
    folder_name: str = DEFAULT_FOLDER_NAME,
    sheet_name: str = DEFAULT_SHEET_NAME,
    folder_id: str | None = None,
    sheet_id: str | None = None,
) -> dict:
    """Create or REUSE the Drive folder and spreadsheet, idempotently.

    A rerun of the helper (for example, to mint a new refresh token) must
    not leave the client with 'NFL Weekly Model (2)' or a second
    spreadsheet. Everything here looks before it creates:

    * the root folder is found by name (drive.file only ever sees this
      app's own files, so a name match is the app's own folder);
    * the ``Model/`` subfolder uses the existing get_or_create helper;
    * the spreadsheet is found by name and type anywhere the app can see,
      so it is rediscovered even if an earlier run crashed before moving
      it into the folder;
    * ``ensure_tabs`` only adds tabs that are missing;
    * the move into the folder is skipped when it already happened.
    """
    from src.google_drive import (
        ensure_file_in_folder,
        ensure_model_folder,
        find_file_by_name,
        get_or_create_root_folder,
    )
    from src.google_sheets import create_spreadsheet, ensure_tabs

    result = {"folder_created": False, "sheet_created": False}

    if folder_id:
        print(f"\nUsing the existing Drive folder {folder_id}.")
    else:
        folder_id, created = get_or_create_root_folder(drive, folder_name)
        result["folder_created"] = created
        if created:
            print(f"\nCreated Drive folder {folder_name!r}.")
        else:
            print(
                f"\nReusing the existing Drive folder {folder_name!r} from a "
                "previous authorization - no duplicate was created."
            )

    ensure_model_folder(drive, folder_id)

    if not sheet_id:
        sheet_id = find_file_by_name(drive, sheet_name, mime_type=SPREADSHEET_MIME)
        if sheet_id:
            print(
                f"Reusing the existing spreadsheet {sheet_name!r} from a "
                "previous authorization - no duplicate was created."
            )
        else:
            sheet_id = create_spreadsheet(sheets, sheet_name)
            result["sheet_created"] = True
            print(f"Created spreadsheet {sheet_name!r}.")
    else:
        print(f"Using the existing spreadsheet {sheet_id}.")

    ensure_tabs(sheets, sheet_id)

    try:
        ensure_file_in_folder(drive, sheet_id, folder_id)
    except Exception as exc:  # the Sheet works either way
        print(f"(Could not move the Sheet into the folder: {exc})")

    result["folder_id"] = folder_id
    result["sheet_id"] = sheet_id
    return result


def format_secrets_block(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    folder_id: str,
    sheet_id: str,
) -> str:
    """The final, copy-ready output. This is the ONLY place the refresh
    token is ever emitted - it is not logged or written to any file."""
    lines = [
        "=" * 72,
        " COPY THESE FIVE VALUES NOW - THEY ARE SHOWN ONLY ONCE",
        "",
        " Add each one as a GitHub Actions repository secret:",
        " Settings -> Secrets and variables -> Actions -> New repository secret",
        "=" * 72,
        f"GOOGLE_CLIENT_ID={client_id}",
        f"GOOGLE_CLIENT_SECRET={client_secret}",
        f"GOOGLE_REFRESH_TOKEN={refresh_token}",
        f"GOOGLE_DRIVE_FOLDER_ID={folder_id}",
        f"GOOGLE_SHEET_ID={sheet_id}",
        "=" * 72,
        " DO NOT CLOSE THIS WINDOW until all five values are copied.",
        " They are secrets: never paste them into email, chat or screenshots.",
        "=" * 72,
    ]
    return "\n".join(lines)


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
        help="Reuse an existing folder instead of discovering/creating one.",
    )
    parser.add_argument(
        "--sheet-id",
        default=None,
        help="Reuse an existing spreadsheet instead of discovering/creating one.",
    )
    parser.add_argument(
        "--scope-drive-full",
        action="store_true",
        help="TROUBLESHOOTING ONLY: request the full Drive scope, which "
        "Google classifies as Restricted. Only needed to write into a "
        "folder the client created by hand in the browser. The normal "
        "production path is the non-sensitive drive.file scope.",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Do not launch a browser; print the sign-in URL to copy by hand.",
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
            "  3. configure the OAuth consent screen and PUBLISH it\n"
            "     ('In production' - Testing tokens die after 7 days)\n"
            "  4. create an OAuth client ID of type 'Desktop app'\n"
            "  5. download the JSON and pass it with --client-secrets\n",
            file=sys.stderr,
        )
        return 2

    scopes = select_scopes(args.scope_drive_full)
    print("Requesting scopes:")
    for scope in scopes:
        print(f"  - {scope}")

    credentials = authorise(args.client_secrets, scopes, args.console)

    from googleapiclient.discovery import build

    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)

    resources = provision(
        drive,
        sheets,
        folder_name=args.folder_name,
        sheet_name=args.sheet_name,
        folder_id=args.drive_folder_id,
        sheet_id=args.sheet_id,
    )
    folder_id = resources["folder_id"]
    sheet_id = resources["sheet_id"]

    client_config = json.loads(args.client_secrets.read_text())
    installed = client_config.get("installed") or client_config.get("web") or {}

    print(
        "\nSpreadsheet:  "
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    )
    print(f"Drive folder: https://drive.google.com/drive/folders/{folder_id}")
    print(
        "\nNEXT STEPS\n"
        "  1. Copy the five values below into GitHub Actions secrets.\n"
        "  2. Delete the client_secret.json file from this machine.\n"
        "  3. Trigger Actions -> NFL Weekly Automation -> Run workflow.\n"
    )

    # The secrets block is deliberately the very last output so it is on
    # screen when the window pauses.
    print(
        format_secrets_block(
            client_id=installed.get("client_id", ""),
            client_secret=installed.get("client_secret", ""),
            refresh_token=credentials.refresh_token,
            folder_id=folder_id,
            sheet_id=sheet_id,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

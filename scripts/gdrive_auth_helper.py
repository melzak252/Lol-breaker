#!/usr/bin/env python3
"""Google Drive OAuth Authentication Helper for Headless DVC Environments.

Exchanges an authorization code or redirect URL for Google Drive OAuth credentials
and saves them to the pydrive2fs cache directory so that `dvc push` / `dvc pull`
can run non-interactively without requiring a local browser on port 8090.
"""

from __future__ import annotations

import os
import sys
import urllib.parse
from pathlib import Path

import appdirs
from pydrive2.auth import GoogleAuth
from pydrive2.fs.spec import COMMON_SETTINGS

CLIENT_ID = "710796635688-iivsgbgsb6uv1fap6635dhvuei09o66c.apps.googleusercontent.com"
CLIENT_SECRET = "a1Fz59uTpVNeG_VGuSKDLJXv"


def get_auth_url() -> str:
    """Generate the exact Google OAuth authorization URL."""
    cache_dir = os.path.join(appdirs.user_cache_dir("pydrive2fs", appauthor=False), CLIENT_ID)
    os.makedirs(cache_dir, exist_ok=True)
    creds_file = os.path.join(cache_dir, "default.json")

    settings = {
        **COMMON_SETTINGS,
        "save_credentials": True,
        "save_credentials_backend": "file",
        "save_credentials_file": creds_file,
        "client_config_backend": "settings",
        "client_config": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "revoke_uri": "https://oauth2.googleapis.com/revoke",
            "redirect_uri": "http://localhost:8090/",
        },
    }
    auth = GoogleAuth(settings=settings)
    return auth.GetAuthUrl()


def save_auth_code(raw_input: str) -> Path:
    """Exchange authorization code or redirect URL and save credentials."""
    raw = raw_input.strip()
    if "code=" in raw:
        parsed = urllib.parse.urlparse(raw)
        qs = urllib.parse.parse_qs(parsed.query)
        if "code" not in qs:
            raise ValueError(f"Could not find 'code' query parameter in URL: {raw}")
        code = qs["code"][0]
    else:
        code = raw

    cache_dir = os.path.join(appdirs.user_cache_dir("pydrive2fs", appauthor=False), CLIENT_ID)
    os.makedirs(cache_dir, exist_ok=True)
    creds_file = os.path.join(cache_dir, "default.json")

    settings = {
        **COMMON_SETTINGS,
        "save_credentials": True,
        "save_credentials_backend": "file",
        "save_credentials_file": creds_file,
        "client_config_backend": "settings",
        "client_config": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "revoke_uri": "https://oauth2.googleapis.com/revoke",
            "redirect_uri": "http://localhost:8090/",
        },
    }

    auth = GoogleAuth(settings=settings)
    auth.GetAuthUrl()  # initialize OAuth2 flow
    auth.Auth(code)    # exchange code for token and save credentials
    return Path(creds_file)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/gdrive_auth_helper.py <AUTH_CODE_OR_REDIRECT_URL>")
        print("\nAuth URL to visit in browser:")
        print(get_auth_url())
        return 1

    try:
        creds_path = save_auth_code(sys.argv[1])
        print(f"Success! Credentials saved to: {creds_path}")
        return 0
    except Exception as e:
        print(f"Error authenticating with Google Drive: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

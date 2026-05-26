"""Simplified version of what
https://github.com/pypa/gh-action-pypi-publish/blob/unstable/v1/oidc-exchange.py
does.
"""

import argparse
import os
import id
import requests


def _repo_url(is_test: bool) -> str:
    if is_test:
        repository_domain = "test.pypi.org"
    else:
        repository_domain = "pypi.org"
    return f"https://{repository_domain}/"


def _token_url(repo_url: str) -> str:
    return f"{repo_url}_/oidc/mint-token"


def _audience(is_test: bool) -> str:
    # these could be fetched using get "_repo_url()/_/oidc/audience"
    # hardcoded is fine
    return "pypi" if not is_test else "testpypi"


def _detect_credential(audience: str, /) -> str:
    token = id.detect_credential(audience=audience)
    if token is None:
        raise id.IdentityError(
            "Attempted to discover OIDC in broken environment",
        )
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Fetches an OIDC token and prints it to stdout"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use the test PyPI repository and audience",
    )
    args = parser.parse_args()
    if not os.getenv("GITHUB_ENV"):
        raise RuntimeError(
            "This script is meant to be run in a GitHub Actions environment."
        )

    repo_url = _repo_url(is_test=args.test)
    token_url = _token_url(repo_url=repo_url)
    audience = _audience(is_test=args.test)

    token = _detect_credential(audience)
    oidc_token_payload: dict[str, str] = {"token": token}
    # Now we can do the actual token exchange.
    mint_token_resp = requests.post(
        token_url,
        json=oidc_token_payload,
        timeout=5,
    )
    mint_token_payload = mint_token_resp.json()
    if not mint_token_resp.ok:
        raise RuntimeError(
            f"Failed to mint token for {repo_url} with audience {audience}. "
            f"Status code: {mint_token_resp.status_code}, response: "
            f"{mint_token_payload}"
        )
    pypi_token = mint_token_payload.get("token")
    assert pypi_token is not None

    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
        f.write(f"TWINE_TOKEN={pypi_token}\n")

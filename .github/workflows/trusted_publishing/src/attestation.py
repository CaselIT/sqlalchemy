import argparse
from pathlib import Path

from pypi_attestations import Attestation, Distribution
from sigstore.models import ClientTrustConfig
from sigstore.oidc import IdentityToken, detect_credential
from sigstore.sign import SigningContext


def main():
    parser = argparse.ArgumentParser(
        description="Creates attestations for the files in the directory"
    )
    parser.add_argument(
        "directory",
        help="The directory containing the files to attest",
    )
    args = parser.parse_args()

    files = [f for f in Path(args.directory).iterdir() if f.is_file()]
    if not files:
        raise RuntimeError(f"No files found in directory {args.directory}")

    oidc_token = detect_credential()
    assert oidc_token is not None, "Failed to detect OIDC token"
    identity = IdentityToken(oidc_token)

    with SigningContext.from_trust_config(ClientTrustConfig.production()).signer(identity, cache=True) as signer:
        for dist_path in files:
            attestation_path = dist_path.with_suffix(dist_path.suffix + ".publish.attestation")
            dist = Distribution.from_file(dist_path)
            attestation = Attestation.sign(signer, dist)

            attestation_path.write_text(attestation.model_dump_json(), encoding='utf-8')
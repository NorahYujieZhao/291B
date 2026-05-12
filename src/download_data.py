"""Download all datasets needed for the Sample Identifiability project (Project 2).

Downloads into ``data/`` (relative to the repository root):

  * ``weightloss_peptidoforms.tsv``       -- MSV000080596 "Peptidoforms intensities"
                                             table (40,921 peptidoforms x 336 samples,
                                             58 patients).  Primary data for the
                                             identifiability task.
  * ``weightloss_variant_coords.tsv``      -- MSV000080596 "Variants amino acid
                                             coordinates per protein" table (used to
                                             map SAAPs to protein positions for the
                                             worldwide-identifiability calculation).
  * ``covid_peptidoforms.tsv``             -- MSV000085507 "Peptidoforms TMT" table
                                             (COVID-19 sera).  Used as an independent
                                             cross-dataset negative reference set.
  * ``SAAP_frequencies_dbSNP_2021.tsv``    -- dbSNP single-amino-acid-polymorphism
                                             population allele frequencies.

The MassIVE/ProteoSAFe tables are produced by clicking the "Download" tab on the
result view pages (see the class project handout).  That button POSTs to the
``DownloadResult`` servlet and returns a zip archive containing a single TSV plus a
``params.xml`` file -- we keep only the TSV.

Run with ``python -m src.download_data`` or ``python src/download_data.py``.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.parse
import urllib.request
import zipfile

# --------------------------------------------------------------------------------------
# Source definitions
# --------------------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# ProteoSAFe "DownloadResult" sources: (host, task id, result view name).
PROTEOSAFE_SOURCES = {
    "weightloss_peptidoforms.tsv": dict(
        host="proteomics3.ucsd.edu",
        task="b11323cd19924f0ea79b0bf8796c80ea",
        view="mq_peptidoforms_intensity",
        note="MSV000080596 Plasma weight loss -- peptidoforms intensities (~173 MB zip)",
    ),
    "weightloss_variant_coords.tsv": dict(
        host="proteomics3.ucsd.edu",
        task="b11323cd19924f0ea79b0bf8796c80ea",
        view="variant_by_protein_region_expression_table",
        note="MSV000080596 Plasma weight loss -- variants amino-acid coords per protein",
    ),
    "covid_peptidoforms.tsv": dict(
        host="proteomics2.ucsd.edu",
        task="81d64e27ee46423cb7824ec8580e53f8",
        view="peptidoform_expression_table",
        note="MSV000085507 COVID-19 sera -- peptidoforms TMT (~20 MB zip)",
    ),
}

# Google-Drive-hosted file: dbSNP SAAP frequency table.
GDRIVE_SOURCES = {
    "SAAP_frequencies_dbSNP_2021.tsv": dict(
        file_id="1m3nZYxnr-Fb1_f2GLkwvRVUouytW4FW-",
        note="SAAP_frequencies_dbSNP_2021.tsv (~115 MB)",
    ),
}

_UA = {"User-Agent": "Mozilla/5.0 (cse291-project2 data downloader)"}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f}GB"


def _read_url(req: urllib.request.Request, *, label: str) -> bytes:
    """Download a URL fully, printing a small progress indicator."""
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        chunks = []
        got = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
            if total:
                pct = 100.0 * got / total
                sys.stdout.write(f"\r  {label}: {_human(got)} / {_human(total)} ({pct:4.1f}%)")
            else:
                sys.stdout.write(f"\r  {label}: {_human(got)}")
            sys.stdout.flush()
        sys.stdout.write("\n")
    return b"".join(chunks)


def _extract_single_tsv(zip_bytes: bytes) -> bytes:
    """Return the bytes of the single ``.tsv`` member of a ProteoSAFe result zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tsv_names = [n for n in zf.namelist() if n.lower().endswith(".tsv")]
        if not tsv_names:
            raise RuntimeError(f"no .tsv inside downloaded archive (members: {zf.namelist()})")
        # Pick the largest TSV in case there is more than one.
        tsv_names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        return zf.read(tsv_names[0])


def _download_proteosafe(dest: str, *, host: str, task: str, view: str, note: str) -> None:
    url = f"https://{host}/ProteoSAFe/DownloadResult?" + urllib.parse.urlencode(
        {"view": view, "task": task}
    )
    body = urllib.parse.urlencode(
        {"option": "delimit", "content": "all", "entries": "", "query": ""}
    ).encode()
    req = urllib.request.Request(url, data=body, headers=_UA)
    print(f"- {os.path.basename(dest)}  [{note}]")
    zip_bytes = _read_url(req, label="downloading zip")
    tsv_bytes = _extract_single_tsv(zip_bytes)
    with open(dest, "wb") as fh:
        fh.write(tsv_bytes)
    print(f"  -> wrote {dest} ({_human(os.path.getsize(dest))})")


def _download_gdrive(dest: str, *, file_id: str, note: str) -> None:
    # The "confirm=t" parameter skips Google Drive's virus-scan interstitial for
    # large files; "drive.usercontent.google.com" is the endpoint that streams bytes.
    url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {"id": file_id, "export": "download", "confirm": "t"}
    )
    req = urllib.request.Request(url, headers=_UA)
    print(f"- {os.path.basename(dest)}  [{note}]")
    data = _read_url(req, label="downloading file")
    if data[:200].lstrip().lower().startswith(b"<!doctype html"):
        raise RuntimeError(
            "Google Drive returned an HTML page instead of the file -- the file id "
            "may be wrong or the file is no longer shared."
        )
    with open(dest, "wb") as fh:
        fh.write(data)
    print(f"  -> wrote {dest} ({_human(os.path.getsize(dest))})")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def download_all(data_dir: str = DATA_DIR, *, force: bool = False) -> dict:
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for name, spec in PROTEOSAFE_SOURCES.items():
        dest = os.path.join(data_dir, name)
        paths[name] = dest
        if os.path.exists(dest) and not force and os.path.getsize(dest) > 0:
            print(f"- {name}: already present ({_human(os.path.getsize(dest))}), skipping")
            continue
        _download_proteosafe(dest, **spec)
    for name, spec in GDRIVE_SOURCES.items():
        dest = os.path.join(data_dir, name)
        paths[name] = dest
        if os.path.exists(dest) and not force and os.path.getsize(dest) > 0:
            print(f"- {name}: already present ({_human(os.path.getsize(dest))}), skipping")
            continue
        _download_gdrive(dest, **spec)
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR, help="destination directory (default: <repo>/data)")
    p.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = p.parse_args(argv)
    download_all(args.data_dir, force=args.force)
    print("\nAll datasets downloaded into", args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

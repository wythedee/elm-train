# Download the complete preprocessed training data

Repository: https://modelscope.cn/datasets/wythedee/elm-data

The `releases/native200_20261101/` transfer contains the entire set of files
referenced by our 16-source historical manifests: 15 LMDB `data.mdb` files and
one HeBin H5, totalling 79,297,912,428 signal bytes, plus manifests and metadata.
There are 447,545 manifest rows: Train313626 / Val61474 / Test72445.
These counts describe the historical preprocessed version; some original
converters capped windows per person. They do not describe every original
recording in the servers' approximately 2.77TB shared dataset directory.

The 15 LMDBs contain the manifest-referenced sample keys plus one or two
metadata entries per database; no large extra sample pool was found there.
The complete HeBin H5 has 127,368 trials, while the historical manifests use
124,892: its additional subject S59 has 2,476 trials. The whole H5 is transferred,
including those trials, while the historical manifests retain their original
57-subject selection. Do not silently add S59 to Train or Test.

Upload is in progress. Wait for the completed release's `dataset_manifest.json`
and the maintainer's completion confirmation before running the full dataset.
A few existing manifests or a successful single-file download do not establish
that all signal files have arrived. This repository is temporarily public for
the owner-authorized collaborator download; the owner plans to make it private
afterward. Keep the original source attribution and terms with the data.

## Download on the training server

Use the ModelScope SDK in a data-management environment, separately from the
pinned training and vLLM environments. The maintainer's upload uses SDK1.37.1;
the command below uses its actual `dataset_snapshot_download` interface.
Allow at least 85GB for the release itself and additional room for feature
caches, checkpoints and temporary files. All paths below are examples.

```bash
export MODELSCOPE_CACHE=/work/elm-private/cache/modelscope
python - <<'PY'
from modelscope.hub.snapshot_download import dataset_snapshot_download
dataset_snapshot_download(
    dataset_id='wythedee/elm-data',
    revision='master',
    local_dir='/work/elm-data',
    allow_patterns=['releases/native200_20261101/**'],
    max_workers=4,
)
PY
```

Public download requires no access token. After the repository becomes private,
use the owner's authorized ModelScope access through a private credential
mechanism; never put a token in Git, a command line or a report.

Set these configuration fields to the actual downloaded paths:

```json
{
  "manifest_root": "/work/elm-data/releases/native200_20261101/manifests",
  "data_root": "/work/elm-data/releases/native200_20261101",
  "path_map": {}
}
```

The released manifests use relative storage paths. They preserve IDs, labels,
channel ordering, scaling, resampling/window settings and historical splits;
`source_file` is reduced to its basename. You do not need to recreate our
server's absolute directories. LMDB lock files are generated locally as needed;
they are not dataset assets.

## Check before training

1. Compare file sizes and per-source/split row counts against the completed
   `dataset_manifest.json` and [data table](DATASETS.md). The release records
   SHA256 values as an optional stronger check.
2. Run `tools/validate_data.py` as described in the [runbook](AGENT_RUNBOOK.md),
   then the pipeline `check` and a bounded `features` stage. Inspect bad records,
   units, channel counts, labels and acceptance coverage rather than silently
   dropping an entire source/class.
3. Preserve the full Val/Test inventories. Historical Mumtaz/NEMAR cohort aliases
   and CHB01/21 cross-split identity conflicts remain; this transfer preserves
   history and does not certify independent generalization. A formal split fix
   must also account for the pretrained Encoder's subject provenance.

Read [preprocessing code and source access](DATASETS.md#preprocessing-code-and-source-access)
if rebuilding from original recordings. Downloading these preprocessed files
does not require rerunning the original raw converters.

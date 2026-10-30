# Pipeline validation and data scope

This document separates software execution from prediction quality. A valid JSON
answer is not necessarily a correct answer. An existing feature cache is not proof
that a new installation can reproduce raw EEG preprocessing.

## Current release boundary

The runnable public entry point, `tools/reproduce.py`, reproduces the archived
80-token Oracle experiment using separately supplied, versioned assets. It does
not currently implement the entire raw-data-to-ELM pipeline. The `reference/`
directory is an archive, not a collection of standalone training commands.

The requested full-data release is being assembled. Do not schedule a large H20
run from this repository until the raw preprocessing adapters, feature extraction,
EEG Projector training, evaluation, and their server acceptance checks are present.
Do not substitute the archived Oracle command for full EEG pretraining.

| Stage | Evidence available | Remaining acceptance work |
|---|---|---|
| Raw data → Native200 | Historical preprocessing and manifests exist | Portable per-dataset adapters and bounded tests from raw input |
| Native200 → frozen Encoder features | Historical feature banks and Encoder checkpoints exist | Configurable checkpoint/input paths and feature reconstruction checks |
| Facts → Oracle tokens → frozen 4B | Archived runnable entry and private-asset CPU checks | New-machine GPU reproduction, including H20 |
| EEG features → Projector → frozen 4B | Private server debug runs have trained, saved, resumed, generated and scored | Publish the current implementation and repeat a bounded test through the public entry |
| Single class/statistic answers | Validation results are available | Full-data evaluation under a corrected cohort split |
| Complete report generation | Structured output is valid; previous checkpoint has content errors | Validate the run with explicit report supervision; report format and factual correctness separately |

All model training and optimizer-based acceptance tests run on authorized Linux
servers. A developer laptop may run syntax, configuration, and synthetic manifest
tests only.

## Three different data sizes

Do not present these as the same experiment:

1. **Historical Native200 manifests:** the larger preprocessed corpus. Some
   adapters capped windows per subject or per subject/label. The reference counts
   in `metadata/native200_reference_counts.json` describe those manifests, not an
   uncapped pass through every raw recording.
2. **Oracle feature bank:** 30,828 Train and 2,048 Val records. Feature preparation
   capped records per logical dataset. The Oracle task pool has 4,691 accepted
   Train questions and 1,744 Val questions; questions are not independent EEG
   participants.
3. **Current EEG Projector debug subset:** 2,326 Train EEG and 128 Val EEG. Asking
   each Val EEG one class question, six statistic questions, and one complete
   report produces 1,024 queries, not 1,024 independent EEG examples.

The 16 logical dataset names include 30 real classes. The 31st Oracle class,
`schizophrenia`, is synthetic only in this experiment. Six statistics are binned
using Train-only thresholds. Shared thresholds, subject identities, label maps,
and preprocessing versions must be fixed before comparing ablations.

## Cohort identity issue found during the release audit

`MUMTAZ_MDD` and `NEMAR_MDD` are two representations of the same source cohort.
The server audit found matching subject identities and exact equality in two
sampled raw EEG segments. Independent splits under these two directory names
therefore do not establish global subject isolation. The current debug subset
contains 12 canonical subjects appearing in Train under one name and Val under
the other. This does not mean all 16 datasets or all Val examples overlap.

Existing debug results remain historical results; their manifests and outputs
have not been rewritten. They are useful for checking information flow and
output formatting, but must not be described as a strict globally subject-disjoint
generalization result. Full-data evaluation requires an explicit cohort policy:

- Retain one representation of this cohort, or explicitly deduplicate it.
- Alternatively, use a common participant assignment across representations and
  declare their sampling weights. Do not count the aliases as independent people.
- Check other known patient/case aliases and overlaps between TUH corpus versions
  before claiming global isolation. Unknown identity mappings remain a limitation.

`tools/validate_data.py` checks the documented Mumtaz/NEMAR alias and rejects
cross-split participants. It does not silently repair existing splits or claim
that every other source has been globally deduplicated.

## Agent check before using supplied manifests

Expected layout:

```text
/work/elm-data/manifests/<DATASET>/train.jsonl
/work/elm-data/manifests/<DATASET>/val.jsonl
/work/elm-data/manifests/<DATASET>/test.jsonl
```

Run with an output directory outside the Git checkout:

```bash
mkdir -p /work/elm-audits
python3 -B tools/validate_data.py \
  --manifest-root /work/elm-data/manifests \
  --output /work/elm-audits/native200_audit.json
```

Default checks require no raw-file SHA: record and subject counts, unique IDs,
labels present, 200 Hz, channel mapping, patch dimensions, and split identity.
Both materialized Native200 LMDB manifests and the explicitly declared historical
HeBin lazy-H5 window variant are supported. The latter crops 1,000 source samples
at 250 Hz starting at sample 125, then resamples to 800 samples at 200 Hz on load;
it is not a missing LMDB file or an interchangeable native-channel EDF adapter.
They do not load LMDB values, check finite waveform samples, or verify label
semantics against the raw annotations. Those are separate adapter acceptance
checks and remain necessary.

Raw file byte identity is optional. Different exports may have different byte
hashes while preserving the same signals. Wrong units, label maps, sample rates,
or participant splits cannot be accepted merely because record counts are close.
Any skipped unreadable records must be listed with counts and reasons.

The validator exits nonzero on a semantic/identity violation or duplicate record
ID, regardless of the permitted malformed-JSON fraction. Its JSON contains subject
identifiers; keep that output with private data rather than committing it.

The current validator was exercised on all 447,545 rows of the historical
16-source manifests on the server: zero schema-invalid rows, and an intentional
nonzero audit result for 27 canonical people spanning multiple splits across
Mumtaz/NEMAR. That count includes Train/Val/Test intersections in the larger
manifests; it is different from the 12 Train/Val overlaps in the debug subset.
Synthetic tests also cover alias normalization, duplicate IDs, and the lazy-H5
variant. This verifies the checker, not the independence of the original data.

## Required full-pipeline acceptance evidence

Before a heavy batch, the released commands must produce evidence for:

1. A small raw recording/annotation example per adapter → expected windows,
   channels, labels, units and subject IDs. Preserve rejection reasons.
2. Manifest → feature extraction with correct padding masks and no gold-label
   leakage into the Projector input. Preserve the existing temporal compression.
3. A short server training run with finite loss, all intended parameter gradients,
   frozen Encoder/Decoder, and checkpoint save/load including optimizer and RNG.
4. Autoregressive generation and independent parser rescoring, including an
   intentionally wrong but syntactically valid report.
5. A report that distinguishes Train/Val/Test, class correctness, six statistic
   scores, full-report all-correct score, format validity, and support counts.
6. A full-data coverage plan: number of unique EEG records and epochs/passes.
   A replacement sampler running many steps is not proof that all Train records
   were visited.

These are acceptance requirements, not claims that the new public pipeline has
already passed them.

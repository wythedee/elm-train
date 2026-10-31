# Pipeline validation and data scope

This document separates software execution from prediction quality. A valid JSON
answer is not necessarily a correct answer. An existing feature cache is not proof
that a new installation can reproduce raw EEG preprocessing.

## Current release boundary

The new entry point is `python -m elm_pipeline.cli`. Its bounded server test has
completed Native200 loading, both frozen Encoders, feature reconstruction, a fresh
30-step Projector run, an independent-process resume to step 35, real 4B
autoregressive generation and CPU report rescoring. A second test started from
ADFTD source files and completed preprocessing → features → generation → report.
Both used authorized A100 servers. These are software acceptance tests.

`tools/reproduce.py` remains the archived 80-token Oracle-only entry point, and
`reference/` remains an archive. The new interface uses the frozen 10200-step
Oracle; it does not yet implement a new full-corpus teacher/Oracle training stage
or training all Encoders from scratch. Do not substitute the archived command for
full EEG pretraining, or launch a heavy H20 batch from these bounded results.

| Stage | Evidence available | Remaining acceptance work |
|---|---|---|
| Raw data → Native200 | Nine adapters included; ADFTD: 9 people, 18 windows, 6/6/6 Train/Val/Test, zero errors | Eight included routes still need raw-input tests; seven other raw routes are not all packaged |
| Native200 → frozen Encoder features | All 16 logical sources, 2 Train + 2 Val records each; 64 fresh records match the old cache within the declared BF16 tolerance, masks/bins exact | Full-corpus extraction, corrected subject identity, each raw-version equivalence |
| Facts → Oracle tokens → frozen 4B | Frozen Oracle positive control generated alongside the Projector | New teacher/Oracle training through this interface; H20 acceptance |
| EEG features → Projector → frozen 4B | New entry trained 0→30 and strictly restored 30→35; all parameters/80 slots received gradients; 512 AR queries and reports completed | Full-data training, scientific ablations and H20 |
| Source EEG → report | ADFTD source → Native200 → both Encoders → existing step-35 Projector/Oracle → 96 AR queries, no extra optimization | Other raw adapters and meaningful trained-model evaluation |
| Exported trained weights → report | Safe step-6600 Projector independently loaded on NORVE; 512 queries on the 32-record subset plus 96 on the raw ADFTD subset, no extra optimization; dependency/coverage/raw-summary checks passed | Full evaluation with matched inputs and corrected data protocol |
| Complete report generation | Closed step-6600 diagnostic: class 66/128, six statistic reads 534/768, report 8/128, format 100% | Accuracy improvement under a valid full-data protocol |

All model training and optimizer-based acceptance tests run on authorized Linux
servers. A developer laptop may run syntax, configuration, and synthetic manifest
tests only.

The 35-step model is freshly initialized and is not the trained step-6600 model.
Its poor prediction accuracy is not evidence that the historical model regressed.
The short test establishes data flow, gradients, strict restore, generation and
report integrity. It does not establish useful EEG decoding accuracy.

The standalone inference test also passed with the trained step-6600 safe export,
separately from the fresh short-training model. Both tests retained the original
closed run and checkpoint. The new 32- and 6-record selections are engineering
subsets, not replacements for the historical 128-record evaluation denominator.

The feature reconstruction check compared 64 selected records by ID: continuous
statistics used `rtol=1e-4, atol=1e-6`, masks and bins matched exactly, all feature
cosines exceeded 0.999. Maximum relative RMS differences were 0.01434 for the
class bank and 0.00553 for the statistics bank. The software test used 32 Train
and 32 Val EEG, visited every selected Train record, and peaked at 13.24 GiB of
Torch allocation. Those numbers do not estimate full-data speed or H20 memory.

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

`tools/validate_data.py` checks the documented Mumtaz/NEMAR and CHB01/CHB21
aliases and rejects cross-split participants. It does not silently repair
existing splits or claim that every other source has been globally deduplicated.

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

The earlier Mumtaz/NEMAR validator was exercised on all 447,545 rows of the historical
16-source manifests on the server: zero schema-invalid rows, and an intentional
nonzero audit result for 27 canonical people spanning multiple splits across
Mumtaz/NEMAR. That count includes Train/Val/Test intersections in the larger
manifests; it is different from the 12 Train/Val overlaps in the debug subset.
Synthetic tests also cover both alias normalizations, duplicate IDs, and the
lazy-H5 variant. This verifies the checker, not the independence of the original
data. A new full audit must be retained when changing the identity policy.

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

Items 2–5 have passed the bounded A100 software test described above. Item 1 has
passed for ADFTD only; item 6 is implemented as a full-pass sampler but has not
been exercised over the entire corpus. Failure handling also refuses a report
with missing/duplicate queries, changed inference dependencies, or a summary
that differs from rescoring the raw strings. Do not expand these statements to
unverified adapters, complete scientific evaluation, or H20 hardware.

`metadata/current_pipeline_provenance.json` identifies the current staged module
and its bounded acceptance scope. The older `metadata/provenance.json` and
`source_manifest.json` identify the archived Oracle source, not this new pipeline.

## vLLM 0.32 inference acceptance, 2026-10-31

The isolated Python 3.11 / Torch 2.13 cu129 / vLLM 0.32 V1 inference profile has
passed real A100 generation and independent reporting against the retained 0.9.2
profile. Each generated 512 fixed Val answers plus four native-ID/embedding
controls. All 512 same-query input embedding hashes matched; 504 predictions
matched. Oracle class/statistic reads were unchanged, while its whole reports
changed from 27/32 correct to 25/32. Preserve this difference rather than claiming
accuracy improvement. See [installation, counts and timing](VLLM_UPGRADE.md).

The training runtime, losses and frozen weights were not changed. No new optimizer
steps or Test generation were performed. This acceptance covers only the fixed
32-EEG software subset; full Val, full-corpus training and H20 remain unverified.

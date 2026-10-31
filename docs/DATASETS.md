# Dataset inventory and validation contract

This table describes the historical Native200 manifests, not a claim that all raw adapters have been verified. Counts are windows in Train / Val / Test. They are not participant counts or uncapped raw-recording coverage. See `metadata/native200_reference_counts.json` for per-split subject counts.

Every source passed a bounded Native200 → two-Encoder reconstruction check on two Train and two Val records. Only ADFTD has also passed a fresh raw-input acceptance test in this release.

|Logical name|Reference windows Train / Val / Test|Raw entry in this release|Version/label issue to check|
|---|---:|---|---|
|ADFTD|2560 / 480 / 480|`scalp_native200` — raw test passed on a bounded subset|EEGLAB eyes-closed derivatives plus participants.tsv, groups A/F/C; actual raw9-person acceptance passed.|
|ATTEN_GONOGO|2304 / 720 / 720|Matching raw route not yet packaged; compatible Native200 supported|Attention/rest semantics and trial boundaries must match the supplied source annotations.|
|CHB_MIT|1328 / 160 / 160|`event_native200` — raw test pending|Bipolar channel source; CHB01/21 same person. Current adapter reproduces legacy case-directory assignments; full identity audit rejects their split conflict.|
|HEBIN2021_NATIVE200_LR|87543 / 17774 / 19575|Matching raw route not yet packaged; compatible Native200 supported|Historical lazy 75-group H5: 250 Hz source, crop start125/length1000, resample to800 samples; explicit file mapping may be needed for a compressed archive.|
|HMC_SLEEP|20878 / 4560 / 4600|`event_native200` — raw test pending|Signal and hypnogram pairs; wake/N1/N2/N3/REM mapping, five-second samples from annotated intervals.|
|MUMTAZ_MDD|4857 / 1166 / 1060|Matching raw route not yet packaged; compatible Native200 supported|Same source cohort as NEMAR; resolve canonical people before mixing.|
|NEMAR_MDD|1676 / 320 / 320|`scalp_native200` — raw test pending|Matching nm000114 derivatives and participant/session metadata; same cohort as Mumtaz.|
|PHYSIONET_4CLASS|12555 / 3429 / 3587|Matching raw route not yet packaged; compatible Native200 supported|Four original motor classes; preserve run-dependent annotation meaning. Do not reduce to left/right.|
|REEG_PD|3183 / 649 / 662|`scalp_native200` — raw test pending|Matching EEGLAB recordings and GROUP field; source/version still to be confirmed with the data owner.|
|SCHIRR_LR|3660 / 1560 / 1520|Matching raw route not yet packaged; compatible Native200 supported|Use the matching left/right trial manifest, not an arbitrary export of the parent multi-class corpus.|
|SEED_BINARY|2970 / 540 / 540|Matching raw route not yet packaged; compatible Native200 supported|Positive/negative subset; neutral is not silently reassigned. Preserve sessions and person identities.|
|SHIN_NATIVE_LR|1200 / 300 / 240|Matching raw route not yet packaged; compatible Native200 supported|Current native30-channel LMDB uses input_scale=.01 and skip_trial_normalization=true; ECHO75-group H5 is a different version.|
|TUAB|76760 / 15360 / 27600|`scalp_native200` — raw test pending|Correct normal/abnormal directories; official evaluation retained as Test, Train subjects split for Val.|
|TUEP|62684 / 10849 / 7421|`scalp_native200` — raw test pending|Epilepsy versus no-epilepsy; inspect source participant/session metadata and TUH cross-corpus IDs.|
|TUEV|2907 / 811 / 2203|`event_native200` — raw test pending|Original six event codes; duplicate channel annotations of the same event are removed.|
|TUSZ|26561 / 2796 / 1757|`event_native200` — raw test pending|Background versus seizure; preserve official train/dev/eval and binary subtype mapping.|

## Per-source acceptance, in execution order

1. Obtain the matching original files and annotations from the source owner or links in `AGENT_RUNBOOK.md`. A dataset name alone is not a version identifier. Record channel naming, units, source sampling rate, label map, subject IDs and sessions before conversion.
2. Run the matching adapter into new output/manifests directories. Keep protocol, caps, rejected recordings and their reasons. For the seven unimplemented raw routes, obtain the exact reader/trial/split builder first; do not tell an Agent to guess a label map or substitute a similar dataset.
3. Compare counts, class support and subject inventory with the reference; explain differences from version/caps/rejections. For scalp adapters, `--max-windows-per-subject 0` means no cap. Event adapters use `--max-windows-per-subject-label 0`. Removing a cap creates a different data version, not a guaranteed exact historical reproduction.
4. Run `tools/validate_data.py`. Canonical aliases and split conflicts are semantic errors, even if every file and row is readable. The CHB raw adapter currently preserves the historical split; it needs an explicit corrected person assignment before formal training. The checker intentionally blocks unsafe full training rather than silently changing history.
5. Inspect each sampled LMDB payload: float32, finite, expected `[channels,patches,200]`, canonical indices matching channel order, correct event/class and actual offset. The validator alone checks manifests, not every waveform or raw annotation.
6. Reconstruct features and retain accepted/rejected counts per source and class. A tolerated broken record cannot erase a selected class or an entire source. Do not force raw byte hashes to match across equivalent exports, but do not accept wrong units/labels/timing to obtain the desired count.
7. Freeze the accepted manifest version, Train-only cuts and Encoder provenance. Keep the full Val/Test inventories fixed. The CLI currently generates Val only; new Test execution is a separate declared stage, not an implicit consequence of a training command.

## Data volume for training

Historical Train has 313626 windows. A shuffled full pass at batch 8 takes 39204 updates and crosses into the next pass to fill the final batch. A 6000-step run contains 48000 sample visits; it is not a full pass of this manifest. The debugger used 32 Train records, while the older scientific Projector diagnostic used 2326. Preserve these three denominators in every report.

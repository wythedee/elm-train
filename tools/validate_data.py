#!/usr/bin/env python3
"""Read-only Native200 manifest audit, including known cross-source cohort aliases.

No model imports, GPU use, preprocessing, rewriting, or automatic split repair.
Raw EEG SHA checks are deliberately not required. This checks manifest semantics;
it does not certify waveform content or every upstream subject identity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
MDD_ALIASES = {"MUMTAZ_MDD", "NEMAR_MDD"}


def canonical_subject(dataset, subject):
    """Do not namespace known aliases into different people by dataset name."""
    if dataset in MDD_ALIASES:
        match = re.fullmatch(r"(?:sub[-_]?)?(h|mdd)[_ -]?s0*(\d+)", subject.lower())
        if match is None:
            raise ValueError("Unknown Mumtaz/NEMAR subject notation: " + repr(subject))
        return "mumtaz_origin:" + match[1] + "_s" + str(int(match[2]))
    if dataset == "CHB_MIT":
        match = re.fullmatch(r"chb0*(\d+)", subject.lower())
        if match is None:
            raise ValueError("Unknown CHB case notation: " + repr(subject))
        # PhysioNet documents chb21 as a later recording of the chb01 person.
        number = int(match[1])
        return "chb_person:" + str(1 if number == 21 else number)
    return dataset + ":" + subject


def audit(manifest_root, datasets, max_bad_fraction=0.0):
    root = Path(manifest_root).resolve(strict=True)
    if not 0 <= max_bad_fraction < 1:
        raise ValueError("max_bad_fraction must be in [0, 1)")
    errors, warnings, summaries = [], [], {}
    identities = defaultdict(lambda: defaultdict(set))
    ids = {}
    total_rows, bad_rows = 0, 0

    def error(kind, **details):
        # Count every violation, retain bounded examples for very large corpora.
        if len(errors) < 200:
            errors.append({"kind": kind, **details})

    critical_count = 0
    for dataset in datasets:
        summaries[dataset] = {}
        for split in SPLITS:
            path = root / dataset / (split + ".jsonl")
            if not path.is_file():
                critical_count += 1
                error("missing_manifest", dataset=dataset, split=split)
                continue
            labels, shapes, rates, subjects = Counter(), Counter(), Counter(), set()
            rows, bad = 0, 0
            with path.open(encoding="utf-8") as stream:
                for lineno, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    total_rows += 1
                    rows += 1
                    try:
                        row = json.loads(line)
                        for key in ("id", "dataset", "subject", "label"):
                            if not isinstance(row.get(key), str) or not row[key].strip():
                                raise ValueError("Missing/non-string " + key)
                        if row["dataset"] != dataset:
                            raise ValueError("Dataset field differs from directory")
                        rate = row.get("sampling_rate")
                        if isinstance(rate, bool) or rate != 200:
                            raise ValueError("Native200 requires sampling_rate=200")
                        shape = row.get("input_shape")
                        if (not isinstance(shape, list) or len(shape) != 2
                                or any(type(v) is not int or v <= 0 for v in shape)):
                            raise ValueError("input_shape must be [channels, samples]")
                        if not 1 <= shape[0] <= 75 or shape[1] % 200:
                            raise ValueError("Unsupported channel count or partial 200-sample patch")
                        indices = row.get("canonical_channel_indices")
                        if (not isinstance(indices, list) or len(indices) != shape[0]
                                or any(type(v) is not int or not 0 <= v < 75 for v in indices)
                                or len(set(indices)) != len(indices)):
                            raise ValueError("Invalid/missing canonical_channel_indices")
                        channels = row.get("channels")
                        # The historical HeBin manifest uses lazy, already-grouped
                        # 75-channel H5 input, not a materialized Native200 LMDB.
                        lazy_hebin = (dataset == "HEBIN2021_NATIVE200_LR"
                                      and row.get("h5_path") and shape[0] == 75
                                      and indices == list(range(75)))
                        if not lazy_hebin and (
                                not isinstance(channels, list) or len(channels) != shape[0]):
                            raise ValueError("channels length differs from input_shape")
                        if row.get("patch_size", 200) != 200:
                            raise ValueError("patch_size differs from 200")
                        if row.get("patch_count", shape[1] // 200) * 200 != shape[1]:
                            raise ValueError("patch_count differs from input_shape")
                        if lazy_hebin:
                            if type(row.get("trial_idx")) is not int or row["trial_idx"] < 0:
                                raise ValueError("Invalid H5 trial_idx")
                            if row.get("resample_on_load") is not True:
                                raise ValueError("HeBin H5 requires declared lazy resampling")
                            if (row.get("source_sampling_rate") != 250
                                    or row.get("window_start") != 125
                                    or row.get("window_length") != 1000
                                    or shape != [75, 800]):
                                raise ValueError("Unsupported HeBin lazy-window variant")
                        elif not row.get("lmdb_path") or not row.get("lmdb_key"):
                            raise ValueError("Native200 LMDB path/key missing")
                        person = canonical_subject(dataset, row["subject"])
                    except (TypeError, ValueError, KeyError) as exc:
                        bad += 1
                        bad_rows += 1
                        # A few unreadable JSON records may be explicitly tolerated.
                        # Semantic/identity errors cannot be relaxed with this flag.
                        if not isinstance(exc, json.JSONDecodeError):
                            critical_count += 1
                        error("invalid_row", dataset=dataset, split=split,
                              line=lineno, reason=str(exc))
                        continue
                    identity = (dataset, row["id"])
                    if identity in ids:
                        critical_count += 1
                        error("duplicate_id", dataset=dataset, id=row["id"],
                              split=split, previous_split=ids[identity])
                    else:
                        ids[identity] = split
                    identities[person][split].add(dataset)
                    subjects.add(person)
                    labels[row["label"]] += 1
                    shapes["x".join(map(str, shape))] += 1
                    rates[str(rate)] += 1
            summaries[dataset][split] = {
                "rows": rows, "invalid_rows": bad, "subjects": len(subjects),
                "label_counts": dict(sorted(labels.items())),
                "input_shapes": dict(shapes), "sampling_rates": dict(rates),
            }
            if not rows or bad == rows:
                critical_count += 1
                error("empty_usable_split", dataset=dataset, split=split)
            if rows and bad / rows > max_bad_fraction:
                critical_count += 1
                error("bad_fraction_exceeded", dataset=dataset, split=split,
                      fraction=bad / rows, allowed=max_bad_fraction)

    overlaps = []
    repeated_people = 0
    for person, by_split in sorted(identities.items()):
        all_datasets = set().union(*by_split.values())
        repeated_people += len(all_datasets) > 1
        if len(by_split) > 1:
            overlaps.append({"cohort_subject": person,
                             "splits": {k: sorted(v) for k, v in by_split.items()}})
    if overlaps:
        critical_count += len(overlaps)
        error("subject_split_overlap", people=len(overlaps))
    if repeated_people:
        warnings.append({"kind": "same_people_multiple_dataset_names",
                         "people": repeated_people,
                         "action": "Declare duplicate cohort weighting or retain one representation."})
    if any(ds in {"TUAB", "TUEP", "TUEV", "TUSZ", "CHB_MIT"} for ds in datasets):
        warnings.append({
            "kind": "additional_identity_mapping_required",
            "action": "Check source patient/case aliases, including across TUH corpus versions. "
                      "This tool merges documented Mumtaz/NEMAR and CHB01/21 aliases; "
                      "a pass does not certify all corpora globally independent."})
    return {
        "status": "passed_manifest_checks" if not critical_count else "failed",
        "datasets": summaries, "total_rows": total_rows, "invalid_rows": bad_rows,
        "max_bad_fraction": max_bad_fraction, "critical_violations": critical_count,
        "errors": errors, "error_examples_limit": 200,
        "cross_split_subject_overlap_count": len(overlaps),
        "cross_split_subject_overlap_examples": overlaps[:100],
        "warnings": warnings, "raw_file_SHA_required": False,
        "waveform_contents_checked": False, "source_data_modified": False,
        "GPU_used": False, "training_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--max-bad-fraction", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    datasets = args.datasets or sorted(
        p.name for p in args.manifest_root.iterdir() if p.is_dir() and (p / "train.jsonl").exists())
    if not datasets:
        parser.error("No dataset manifests found")
    if args.output:
        output = args.output.resolve()
        if output == REPO or REPO in output.parents:
            parser.error("Keep per-subject audit output outside the code repository")
        if output.exists():
            parser.error("Refusing to overwrite an existing audit")
        if not output.parent.is_dir():
            parser.error("Create the output parent directory first")
    result = audit(args.manifest_root, datasets, args.max_bad_fraction)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({k: result[k] for k in (
            "status", "total_rows", "invalid_rows", "critical_violations",
            "cross_split_subject_overlap_count")}, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed_manifest_checks" else 1


if __name__ == "__main__":
    sys.exit(main())

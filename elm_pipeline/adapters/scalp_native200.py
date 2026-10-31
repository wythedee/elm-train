"""Convert newly downloaded scalp EEG datasets to the mainline Native200 contract."""
from __future__ import annotations
import hashlib
import argparse
import csv
import json
import os
import pickle
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import lmdb
import mne
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from elm_pipeline.channels import CANONICAL_CHANNELS, canonical_channel_indices
TARGET_RATE = 200
WINDOW_SECONDS = 5
WINDOW_SAMPLES = TARGET_RATE * WINDOW_SECONDS
PATCH_SIZE = TARGET_RATE
PATCH_COUNT = WINDOW_SECONDS
DEFAULT_SEED = 8231
DEFAULT_OUTPUT_ROOT = None
DEFAULT_MANIFEST_ROOT = None

@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    task: str
    subject: str
    session: str
    label: str
    path: Path
    fixed_split: str | None = None
DATASET_ROOTS = {'ADFTD': None, 'REEG_PD': None, 'NEMAR_MDD': None, 'TUAB': None, 'TUEP': None}
DATASET_NOTCH_HZ = {'TUAB': 60.0, 'TUEP': 60.0, 'TUEV': 60.0, 'TUSZ': 60.0}

def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def discover_adftd(root: Path) -> list[SourceRecord]:
    label_map = {'A': 'alzheimers_disease', 'F': 'frontotemporal_dementia', 'C': 'healthy_control'}
    labels = {row['participant_id']: label_map[row['Group'].strip()] for row in read_tsv(root / 'participants.tsv')}
    records = []
    for path in sorted((root / 'derivatives').glob('sub-*/eeg/*task-eyesclosed_eeg.set')):
        subject = path.parts[-3]
        records.append(SourceRecord('ADFTD', 'dementia_classification', subject, 'eyes_closed', labels[subject], path))
    return records

def discover_reeg_pd(root: Path) -> list[SourceRecord]:
    labels = {}
    for row in read_tsv(root / 'participants.tsv'):
        group = row['GROUP'].strip().lower()
        labels[row['participant_id']] = 'parkinsons_disease' if group == 'pd' else 'healthy_control'
    records = []
    for path in sorted(root.glob('sub-*/eeg/*.set')):
        subject = path.parts[-3]
        records.append(SourceRecord('REEG_PD', 'parkinsons_disease', subject, 'eyes_open', labels[subject], path))
    return records

def discover_nemar_mdd(root: Path) -> list[SourceRecord]:
    records = []
    for path in sorted(root.glob('sub-*/eeg/*task-eyesClosed_eeg.edf')):
        subject = path.parts[-3]
        label = 'major_depressive_disorder' if subject.lower().startswith('sub-mdds') else 'healthy_control'
        records.append(SourceRecord('NEMAR_MDD', 'major_depressive_disorder', subject, 'eyes_closed', label, path))
    return records

def discover_sandiego_pd(root: Path) -> list[SourceRecord]:
    records = []
    for path in sorted(root.glob('sub-*/ses-*/eeg/*.bdf')):
        subject = path.parts[-4]
        session = path.parts[-3]
        is_pd = subject.lower().startswith('sub-pd')
        if is_pd and session != 'ses-off' or (not is_pd and session != 'ses-hc'):
            continue
        label = 'parkinsons_disease' if is_pd else 'healthy_control'
        records.append(SourceRecord('SANDIEGO_PD', 'parkinsons_disease', subject, session.removeprefix('ses-'), label, path))
    return records

def discover_nmt(root: Path) -> list[SourceRecord]:
    with (root / 'Labels.csv').open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        label = row['label'].strip().lower()
        location = row['loc'].strip().lower()
        path = root / label / location / row['recordname']
        records.append(SourceRecord('NMT_ABNORMAL', 'eeg_abnormality', Path(row['recordname']).stem, location, label, path, fixed_split='test' if location == 'eval' else None))
    return records

def discover_schizo(root: Path) -> list[SourceRecord]:
    records = []
    for path in sorted(root.glob('*.edf')):
        subject = path.stem.lower()
        label = 'healthy_control' if subject.startswith('h') else 'schizophrenia'
        records.append(SourceRecord('SCHIZO', 'schizophrenia', subject, 'rest', label, path))
    return records

def discover_tuab(root: Path) -> list[SourceRecord]:
    records = []
    for (split_name, fixed_split) in (('train', None), ('eval', 'test')):
        for (label_dir, label) in (('normal', 'normal'), ('abnormal', 'abnormal')):
            base = root / 'edf' / split_name / label_dir / '01_tcp_ar'
            for path in sorted(base.glob('*.edf')):
                subject = path.stem.split('_', 1)[0]
                records.append(SourceRecord('TUAB', 'eeg_abnormality', subject, path.stem, label, path, fixed_split=fixed_split))
    return records

def discover_tuep(root: Path) -> list[SourceRecord]:
    records = []
    label_roots = (('epilepsy_edf', 'epilepsy'), ('no_epilepsy_edf', 'no_epilepsy'))
    for (directory, label) in label_roots:
        base = root / directory
        for path in sorted(base.glob('*/*/*/*.edf')):
            subject = path.relative_to(base).parts[0]
            records.append(SourceRecord('TUEP', 'epilepsy_diagnosis', subject, path.stem, label, path))
    return records
DISCOVERERS = {'ADFTD': discover_adftd, 'REEG_PD': discover_reeg_pd, 'NEMAR_MDD': discover_nemar_mdd, 'SANDIEGO_PD': discover_sandiego_pd, 'NMT_ABNORMAL': discover_nmt, 'SCHIZO': discover_schizo, 'TUAB': discover_tuab, 'TUEP': discover_tuep}

def stratified_subject_split(records: list[SourceRecord], seed: int) -> dict[str, str]:
    fixed = {record.subject: record.fixed_split for record in records if record.fixed_split}
    candidates: dict[str, list[str]] = defaultdict(list)
    subject_label = {}
    for record in records:
        subject_label.setdefault(record.subject, record.label)
    for (subject, label) in subject_label.items():
        if subject not in fixed:
            candidates[label].append(subject)
    assignments = dict(fixed)
    for (label_index, (label, subjects)) in enumerate(sorted(candidates.items())):
        rng = random.Random(seed + label_index * 1009)
        subjects = sorted(subjects)
        rng.shuffle(subjects)
        count = len(subjects)
        if count < 3:
            raise ValueError(f'label {label!r} has only {count} non-fixed subjects')
        test_count = max(1, round(count * 0.15))
        val_count = max(1, round(count * 0.15))
        if test_count + val_count >= count:
            val_count = 1
            test_count = 1
        for subject in subjects[:test_count]:
            assignments[subject] = 'test'
        for subject in subjects[test_count:test_count + val_count]:
            assignments[subject] = 'val'
        for subject in subjects[test_count + val_count:]:
            assignments[subject] = 'train'
    return assignments

def normalize_channel_name(value: str) -> str:
    name = value.strip()
    name = re.sub('^EEG[\\s_-]+', '', name, flags=re.IGNORECASE)
    name = re.sub('[-_](LE|REF|AVG|AR)$', '', name, flags=re.IGNORECASE)
    name = re.sub('[-_](M1|M2|A1|A2)$', '', name, flags=re.IGNORECASE)
    name = name.strip(' ._')
    lookup = {channel.upper(): channel for channel in CANONICAL_CHANNELS}
    aliases = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6', 'M1': 'A1', 'M2': 'A2'}
    normalized = aliases.get(name.upper(), name.upper())
    return lookup.get(normalized, name)

def read_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == '.set':
        return mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
    if suffix == '.bdf':
        return mne.io.read_raw_bdf(path, preload=True, verbose='ERROR')
    if suffix == '.edf':
        return mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    raise ValueError(f'unsupported EEG format: {path}')

def select_canonical_eeg(raw: mne.io.BaseRaw, minimum_channels: int=8) -> tuple[mne.io.BaseRaw, list[str], list[int]]:
    eeg_indices = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False, stim=False, exclude=[])
    if len(eeg_indices) == 0:
        eeg_indices = np.arange(len(raw.ch_names))
    selected_indices = []
    names = []
    canonical_indices = []
    occupied = set()
    for index in eeg_indices.tolist():
        name = normalize_channel_name(raw.ch_names[index])
        canonical_index = canonical_channel_indices([name])[0]
        if canonical_index < 0 or canonical_index in occupied:
            continue
        occupied.add(canonical_index)
        selected_indices.append(index)
        names.append(CANONICAL_CHANNELS[canonical_index])
        canonical_indices.append(canonical_index)
    if len(selected_indices) < minimum_channels:
        raise ValueError(f'only {len(selected_indices)} channels map to the canonical montage: {raw.ch_names}')
    raw.pick(selected_indices)
    return (raw, names, canonical_indices)

def preprocess_raw(record: SourceRecord) -> tuple[np.ndarray, list[str], list[int], float]:
    raw = read_raw(record.path)
    source_rate = float(raw.info['sfreq'])
    (raw, channels, channel_indices) = select_canonical_eeg(raw)
    raw.filter(l_freq=0.3, h_freq=75.0, method='fir', n_jobs=1, verbose='ERROR')
    notch_hz = DATASET_NOTCH_HZ.get(record.dataset, 50.0)
    raw.notch_filter(freqs=[notch_hz], method='fir', n_jobs=1, verbose='ERROR')
    if round(source_rate) != TARGET_RATE:
        raw.resample(TARGET_RATE, npad='auto', n_jobs=1, verbose='ERROR')
    data = raw.get_data().astype(np.float32, copy=False) * np.float32(1000000.0)
    del raw
    return (data, channels, channel_indices, source_rate)

def window_starts(sample_count: int, max_windows: int) -> np.ndarray:
    available = sample_count // WINDOW_SAMPLES
    if available <= 0:
        return np.empty(0, dtype=np.int64)
    if max_windows <= 0 or available <= max_windows:
        return np.arange(available, dtype=np.int64) * WINDOW_SAMPLES
    selected = np.linspace(0, available - 1, num=max_windows, dtype=np.int64)
    return np.unique(selected) * WINDOW_SAMPLES

def safe_id(value: str) -> str:
    return re.sub('[^a-zA-Z0-9]+', '_', value).strip('_').lower()

def window_record_id(dataset, record, root, start):
    """Identify the actual window independently of cap/index/machine root."""
    source = hashlib.sha256(str(record.path.relative_to(root)).encode()).hexdigest()[:16]
    return f'{safe_id(dataset)}_{safe_id(record.subject)}_{safe_id(record.session)}_{source}_start{int(start):010d}_n{WINDOW_SAMPLES}'

def build_dataset(dataset: str, output_root: Path, manifest_root: Path, max_windows_per_subject: int, seed: int, max_subjects: int | None, overwrite: bool) -> dict:
    root = DATASET_ROOTS[dataset]
    records = DISCOVERERS[dataset](root)
    missing = [str(record.path) for record in records if not record.path.is_file()]
    if missing:
        raise FileNotFoundError(f'{dataset}: {len(missing)} source files are missing; first={missing[0]}')
    if max_subjects is not None:
        subject_records: dict[str, list[SourceRecord]] = defaultdict(list)
        strata: dict[tuple[str, str], list[str]] = defaultdict(list)
        for record in records:
            subject_records[record.subject].append(record)
        for (subject, subject_items) in subject_records.items():
            item = subject_items[0]
            strata[item.label, item.fixed_split or 'development'].append(subject)
        chosen = set()
        ordered_strata = sorted(strata.items())
        while len(chosen) < min(max_subjects, len(subject_records)):
            changed = False
            for (_, subjects) in ordered_strata:
                for subject in sorted(subjects):
                    if subject not in chosen:
                        chosen.add(subject)
                        changed = True
                        break
                if len(chosen) >= max_subjects:
                    break
            if not changed:
                break
        records = [record for record in records if record.subject in chosen]
    splits = stratified_subject_split(records, seed)
    final_lmdb = output_root / dataset / 'processed_average'
    final_manifests = manifest_root / dataset
    complete_marker = final_manifests / 'protocol.json'
    if complete_marker.is_file() and (not overwrite):
        current = json.loads(complete_marker.read_text())
        if current.get('status') == 'complete':
            print(f'[{dataset}] already complete; skipping', flush=True)
            return current
    suffix = f'.tmp.{os.getpid()}'
    tmp_lmdb = Path(str(final_lmdb) + suffix)
    tmp_manifests = Path(str(final_manifests) + suffix)
    shutil.rmtree(tmp_lmdb, ignore_errors=True)
    shutil.rmtree(tmp_manifests, ignore_errors=True)
    tmp_lmdb.mkdir(parents=True)
    tmp_manifests.mkdir(parents=True)
    env = lmdb.open(str(tmp_lmdb), map_size=256 * 1024 ** 3, subdir=True, lock=True, writemap=False, map_async=True)
    keys_by_split: dict[str, list[bytes]] = {split: [] for split in ('train', 'val', 'test')}
    rows_by_split: dict[str, list[dict]] = {split: [] for split in keys_by_split}
    errors = []
    label_ids = {label: index for (index, label) in enumerate(sorted({record.label for record in records}))}
    source_rates = Counter()
    observed_channels: dict[tuple[str, ...], int] = Counter()
    try:
        for (record_index, record) in enumerate(records, start=1):
            split = splits[record.subject]
            try:
                (data, channels, channel_indices, source_rate) = preprocess_raw(record)
                starts = window_starts(data.shape[-1], max_windows_per_subject)
                if starts.size == 0:
                    raise ValueError(f'record has fewer than {WINDOW_SAMPLES} samples after resampling')
                source_rates[str(source_rate)] += 1
                observed_channels[tuple(channels)] += 1
                with env.begin(write=True) as txn:
                    for (window_index, start) in enumerate(starts.tolist()):
                        sample = np.ascontiguousarray(data[:, start:start + WINDOW_SAMPLES]).reshape(len(channels), PATCH_COUNT, PATCH_SIZE)
                        key_text = window_record_id(dataset, record, root, start)
                        key = key_text.encode('utf-8')
                        payload = {'sample': sample, 'label': record.label, 'raw_label': label_ids[record.label], 'subject': record.subject, 'session': record.session, 'source_id': record.path.name}
                        if not txn.put(key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), overwrite=False):
                            raise RuntimeError(f'duplicate LMDB key: {key_text}')
                        keys_by_split[split].append(key)
                        rows_by_split[split].append({'canonical_channel_indices': channel_indices, 'channels': channels, 'dataset': dataset, 'id': key_text, 'input_scale': 1.0, 'input_shape': [len(channels), WINDOW_SAMPLES], 'label': record.label, 'lmdb_key': key_text, 'lmdb_path': str(final_lmdb), 'patch_count': PATCH_COUNT, 'patch_size': PATCH_SIZE, 'raw_label': label_ids[record.label], 'sampling_rate': TARGET_RATE, 'session': record.session, 'skip_trial_normalization': False, 'source_file': str(record.path.relative_to(root)), 'source_sampling_rate': source_rate, 'subject': record.subject, 'task': record.task, 'source_window_start_sample': int(start), 'source_window_length_samples': WINDOW_SAMPLES, 'legacy_window_index': int(window_index), 'preprocessing_version': 'native200_offset_ids_v1'})
                print(f'[{dataset}] {record_index}/{len(records)} {record.subject}: {len(channels)}ch, {len(starts)} windows', flush=True)
            except Exception as exc:
                errors.append({'subject': record.subject, 'path': str(record.path), 'error': repr(exc)})
                print(f'[{dataset}] ERROR {record.subject}: {exc!r}', file=sys.stderr, flush=True)
        if not any(keys_by_split.values()):
            raise RuntimeError(f'{dataset}: no samples were produced')
        with env.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys_by_split, protocol=pickle.HIGHEST_PROTOCOL))
        env.sync()
    finally:
        env.close()
    for (split, rows) in rows_by_split.items():
        with (tmp_manifests / f'{split}.jsonl').open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + '\n')
    split_summary = {}
    for (split, rows) in rows_by_split.items():
        subjects = sorted({row['subject'] for row in rows})
        split_summary[split] = {'participants': subjects, 'participant_count': len(subjects), 'samples': len(rows), 'label_distribution': dict(sorted(Counter((row['label'] for row in rows)).items()))}
    split_subjects = {split: set(summary['participants']) for (split, summary) in split_summary.items()}
    overlap = {'train_val': sorted(split_subjects['train'] & split_subjects['val']), 'train_test': sorted(split_subjects['train'] & split_subjects['test']), 'val_test': sorted(split_subjects['val'] & split_subjects['test'])}
    if any(overlap.values()):
        raise RuntimeError(f'subject leakage detected: {overlap}')
    protocol = {'dataset': dataset, 'status': 'complete', 'source': str(root), 'source_files': len(records), 'participants': len({record.subject for record in records}), 'sampling_rate': TARGET_RATE, 'window_seconds': WINDOW_SECONDS, 'patch_size': PATCH_SIZE, 'patch_count': PATCH_COUNT, 'max_windows_per_subject': max_windows_per_subject, 'labels': label_ids, 'source_sampling_rates': dict(sorted(source_rates.items())), 'preprocessing': {'channel_selection': 'channels mapped to the mainline 75-channel canonical montage', 'resample_hz': TARGET_RATE, 'bandpass_hz': [0.3, 75.0], 'notch_hz': DATASET_NOTCH_HZ.get(dataset, 50.0), 'stored_unit': 'uV', 'window_normalization': 'deferred to the training loader', 'window_sampling': 'non-overlapping windows uniformly selected over each recording'}, 'split_policy': 'official eval is test and official train is stratified into train/val' if dataset in {'NMT_ABNORMAL', 'TUAB'} else 'stratified subject-disjoint 70/15/15 split', 'seed': seed, 'splits': split_summary, 'subject_overlap': overlap, 'failed_records': errors, 'channel_montages': [{'channels': list(channels), 'record_count': count} for (channels, count) in sorted(observed_channels.items(), key=lambda item: (-item[1], item[0]))], 'id_policy': 'relative-file digest + actual 200-Hz start sample and length; stable across window caps'}
    (tmp_manifests / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=True) + '\n')
    (tmp_lmdb / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=True) + '\n')
    if final_lmdb.exists():
        if not overwrite:
            raise FileExistsError(final_lmdb)
        shutil.rmtree(final_lmdb)
    if final_manifests.exists():
        if not overwrite:
            raise FileExistsError(final_manifests)
        shutil.rmtree(final_manifests)
    final_lmdb.parent.mkdir(parents=True, exist_ok=True)
    final_manifests.parent.mkdir(parents=True, exist_ok=True)
    tmp_lmdb.rename(final_lmdb)
    tmp_manifests.rename(final_manifests)
    print(f'[{dataset}] complete: ' + ', '.join((f'{split}={len(rows)}' for (split, rows) in rows_by_split.items())) + f', errors={len(errors)}', flush=True)
    return protocol

def validate_dataset(dataset: str, output_root: Path, manifest_root: Path) -> None:
    lmdb_path = output_root / dataset / 'processed_average'
    protocol_path = manifest_root / dataset / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['status'] == 'complete'
    assert not any(protocol['subject_overlap'].values())
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, subdir=True)
    try:
        with env.begin() as txn:
            keys_by_split = pickle.loads(txn.get(b'__keys__'))
            for split in ('train', 'val', 'test'):
                assert len(keys_by_split[split]) == protocol['splits'][split]['samples']
                for key in keys_by_split[split][:min(3, len(keys_by_split[split]))]:
                    payload = pickle.loads(txn.get(key))
                    sample = payload['sample']
                    assert sample.dtype == np.float32
                    assert sample.ndim == 3 and sample.shape[1:] == (PATCH_COUNT, PATCH_SIZE)
                    assert np.isfinite(sample).all()
    finally:
        env.close()
    print(f'[{dataset}] validation passed', flush=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=sorted(DATASET_ROOTS), default=sorted(DATASET_ROOTS))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifest-root', type=Path, required=True)
    parser.add_argument('--max-windows-per-subject', type=int, default=40)
    parser.add_argument('--max-subjects', type=int)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--raw-roots', type=Path, required=True)
    args = parser.parse_args()
    locations = json.loads(args.raw_roots.read_text())
    for dataset in args.datasets:
        path = Path(locations[dataset]).expanduser()
        if not path.is_absolute():
            path = args.raw_roots.resolve().parent / path
        DATASET_ROOTS[dataset] = path.resolve(strict=True)
    if args.output_root.exists() or args.manifest_root.exists():
        raise FileExistsError('Use NEW output and manifest roots; old data and partials are preserved')
    args.overwrite = False
    return args

def main() -> None:
    args = parse_args()
    mne.set_log_level('ERROR')
    for dataset in args.datasets:
        build_dataset(dataset=dataset, output_root=args.output_root, manifest_root=args.manifest_root, max_windows_per_subject=args.max_windows_per_subject, seed=args.seed, max_subjects=args.max_subjects, overwrite=args.overwrite)
        validate_dataset(dataset, args.output_root, args.manifest_root)
if __name__ == '__main__':
    main()

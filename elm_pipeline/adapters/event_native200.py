"""Convert event-labelled BrainCoT scalp datasets to the Native200 contract."""
from __future__ import annotations
import argparse
import csv
import hashlib
import heapq
import json
import os
import pickle
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import lmdb
import mne
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from elm_pipeline.adapters.scalp_native200 import CANONICAL_CHANNELS, PATCH_COUNT, PATCH_SIZE, TARGET_RATE, WINDOW_SAMPLES, WINDOW_SECONDS, preprocess_raw, safe_id, select_canonical_eeg
DEFAULT_OUTPUT_ROOT = None
DEFAULT_MANIFEST_ROOT = None
DEFAULT_SEED = 9041
DATASET_ROOTS = {'TUEV': None, 'TUSZ': None, 'HMC_SLEEP': None, 'CHB_MIT': None}
TASKS = {'TUEV': 'eeg_event_classification', 'TUSZ': 'seizure_detection', 'HMC_SLEEP': 'sleep_staging', 'CHB_MIT': 'seizure_detection'}
TUEV_LABELS = {1: 'spsw', 2: 'gped', 3: 'pled', 4: 'eyem', 5: 'artf', 6: 'bckg'}
HMC_LABELS = {'Sleep stage W': 'wake', 'Sleep stage N1': 'n1', 'Sleep stage N2': 'n2', 'Sleep stage N3': 'n3', 'Sleep stage N4': 'n3', 'Sleep stage R': 'rem'}
CHB_CHANNEL_MAP = {'FP1-F7': 'Fp1', 'F7-T7': 'FT7', 'T7-P7': 'TP7', 'P7-O1': 'PO7', 'FP2-F8': 'Fp2', 'F8-T8': 'FT8', 'T8-P8': 'TP8', 'P8-O2': 'PO8', 'FP1-F3': 'F3', 'F3-C3': 'C3', 'C3-P3': 'CP3', 'P3-O1': 'PO3', 'FP2-F4': 'F4', 'F4-C4': 'C4', 'C4-P4': 'CP4', 'P4-O2': 'PO4'}
CANONICAL_INDEX = {name: index for (index, name) in enumerate(CANONICAL_CHANNELS)}

@dataclass(frozen=True)
class Candidate:
    dataset: str
    subject: str
    session: str
    label: str
    path: Path
    start_seconds: float
    split_group: str

def interval_starts(start: float, stop: float) -> Iterator[float]:
    duration = stop - start
    if duration <= 0:
        return
    if duration < WINDOW_SECONDS:
        yield max(0.0, (start + stop - WINDOW_SECONDS) / 2.0)
        return
    position = start
    final_start = stop - WINDOW_SECONDS
    while position <= final_start + 1e-06:
        yield position
        position += WINDOW_SECONDS

def discover_tuev(root: Path) -> Iterator[Candidate]:
    for (source_split, split_group) in (('train', 'train_pool'), ('eval', 'test')):
        split_root = root / 'edf' / source_split
        for path in sorted(split_root.glob('*/*.edf')):
            rec_path = path.with_suffix('.rec')
            if not rec_path.is_file():
                continue
            subject = path.parent.name
            seen = set()
            with rec_path.open('r', encoding='utf-8-sig', newline='') as handle:
                for row in csv.reader(handle):
                    if len(row) < 4:
                        continue
                    try:
                        (start, stop, label_id) = (float(row[1]), float(row[2]), int(float(row[3])))
                    except ValueError:
                        continue
                    label = TUEV_LABELS.get(label_id)
                    if label is None:
                        continue
                    identity = (round(start, 4), round(stop, 4), label)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    center_start = max(0.0, (start + stop - WINDOW_SECONDS) / 2.0)
                    yield Candidate('TUEV', subject, path.stem, label, path, center_start, split_group)

def discover_tusz(root: Path) -> Iterator[Candidate]:
    split_map = {'train': 'train', 'dev': 'val', 'eval': 'test'}
    for (source_split, split_group) in split_map.items():
        split_root = root / 'edf' / source_split
        for csv_path in sorted(split_root.glob('**/*.csv_bi')):
            edf_path = csv_path.with_suffix('').with_suffix('.edf')
            if not edf_path.is_file():
                continue
            subject = csv_path.relative_to(split_root).parts[0]
            seen = set()
            with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
                rows = (line for line in handle if not line.startswith('#'))
                for row in csv.DictReader(rows, skipinitialspace=True):
                    try:
                        start = float(row['start_time'])
                        stop = float(row['stop_time'])
                    except (KeyError, TypeError, ValueError):
                        continue
                    raw_label = str(row.get('label', '')).strip().lower()
                    label = 'background' if raw_label == 'bckg' else 'seizure'
                    for window_start in interval_starts(start, stop):
                        identity = (round(window_start, 4), label)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        yield Candidate('TUSZ', subject, edf_path.stem, label, edf_path, window_start, split_group)

def discover_hmc(root: Path) -> Iterator[Candidate]:
    recordings = root / 'HMC' / 'physionet.org' / 'files' / 'hmc-sleep-staging' / '1.1' / 'recordings'
    for score_path in sorted(recordings.glob('SN*_sleepscoring.txt')):
        subject = score_path.name.split('_', 1)[0]
        edf_path = recordings / f'{subject}.edf'
        if not edf_path.is_file():
            continue
        with score_path.open('r', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle, skipinitialspace=True):
                annotation = str(row.get('Annotation', '')).strip()
                label = HMC_LABELS.get(annotation)
                if label is None:
                    continue
                try:
                    start = float(row['Recording onset'])
                    stop = start + float(row['Duration'])
                except (KeyError, TypeError, ValueError):
                    continue
                for window_start in interval_starts(start, stop):
                    yield Candidate('HMC_SLEEP', subject, subject, label, edf_path, window_start, 'all')

def chb_split(subject: str) -> str:
    number = int(subject.removeprefix('chb'))
    if number <= 20:
        return 'train'
    if number <= 22:
        return 'val'
    return 'test'

def discover_chb(root: Path) -> Iterator[Candidate]:
    for path in sorted((root / 'tmp').glob('chb*/*.pkl')):
        subject = path.parent.name
        try:
            with path.open('rb') as handle:
                payload = pickle.load(handle)
            arrays = [value for (key, value) in payload.items() if key != 'metadata']
            if not arrays:
                continue
            sample_count = len(arrays[0])
            source_rate = 256.0
            intervals = [(max(0, int(start)) / source_rate, max(0, int(stop)) / source_rate) for (start, stop) in payload.get('metadata', {}).get('times', [])]
            del payload, arrays
        except Exception as exc:
            print(f'[CHB_MIT] metadata error {path}: {exc!r}', file=sys.stderr, flush=True)
            continue
        duration = sample_count / source_rate
        position = 0.0
        while position + WINDOW_SECONDS <= duration + 1e-06:
            stop = position + WINDOW_SECONDS
            seizure = any((not (stop <= left or position >= right) for (left, right) in intervals))
            label = 'seizure' if seizure else 'background'
            yield Candidate('CHB_MIT', subject, path.stem, label, path, position, chb_split(subject))
            position += WINDOW_SECONDS
DISCOVERERS = {'TUEV': discover_tuev, 'TUSZ': discover_tusz, 'HMC_SLEEP': discover_hmc, 'CHB_MIT': discover_chb}

def candidate_score(candidate: Candidate, seed: int) -> int:
    value = '|'.join((str(seed), candidate.dataset, candidate.subject, candidate.label, str(candidate.path), f'{candidate.start_seconds:.6f}'))
    return int.from_bytes(hashlib.blake2b(value.encode('utf-8'), digest_size=8).digest(), 'big')

def capped_candidates(candidates: Iterable[Candidate], cap: int, seed: int) -> list[Candidate]:
    heaps: dict[tuple[str, str], list[tuple[int, int, Candidate]]] = defaultdict(list)
    sequence = 0
    for candidate in candidates:
        key = (candidate.subject, candidate.label)
        score = candidate_score(candidate, seed)
        item = (-score, sequence, candidate)
        sequence += 1
        heap = heaps[key]
        if cap <= 0:
            heapq.heappush(heap, item)
        elif len(heap) < cap:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [item[2] for heap in heaps.values() for item in heap]
    return sorted(selected, key=lambda item: (item.subject, item.label, str(item.path), item.start_seconds))

def assign_splits(dataset: str, candidates: list[Candidate], seed: int) -> dict[str, str]:
    subjects = sorted({candidate.subject for candidate in candidates})
    if dataset in {'TUSZ', 'CHB_MIT'}:
        assignment = {}
        for candidate in candidates:
            previous = assignment.setdefault(candidate.subject, candidate.split_group)
            if previous != candidate.split_group:
                raise RuntimeError(f'subject {candidate.subject} appears in multiple official splits')
        return assignment
    if dataset == 'TUEV':
        test_subjects = {candidate.subject for candidate in candidates if candidate.split_group == 'test'}
        pool = sorted(set(subjects) - test_subjects)
        rng = random.Random(seed)
        rng.shuffle(pool)
        val_count = max(1, round(len(pool) * 0.2))
        val_subjects = set(pool[:val_count])
        return {subject: 'test' if subject in test_subjects else 'val' if subject in val_subjects else 'train' for subject in subjects}
    rng = random.Random(seed)
    rng.shuffle(subjects)
    test_count = max(1, round(len(subjects) * 0.15))
    val_count = max(1, round(len(subjects) * 0.15))
    test_subjects = set(subjects[:test_count])
    val_subjects = set(subjects[test_count:test_count + val_count])
    return {subject: 'test' if subject in test_subjects else 'val' if subject in val_subjects else 'train' for subject in subjects}

class NativeWriter:

    def __init__(self, dataset: str, output_root: Path, manifest_root: Path, overwrite: bool):
        self.dataset = dataset
        self.final_lmdb = output_root / dataset / 'processed_average'
        self.final_manifests = manifest_root / dataset
        suffix = f'.tmp.{os.getpid()}'
        self.tmp_lmdb = Path(str(self.final_lmdb) + suffix)
        self.tmp_manifests = Path(str(self.final_manifests) + suffix)
        shutil.rmtree(self.tmp_lmdb, ignore_errors=True)
        shutil.rmtree(self.tmp_manifests, ignore_errors=True)
        self.tmp_lmdb.mkdir(parents=True)
        self.tmp_manifests.mkdir(parents=True)
        self.env = lmdb.open(str(self.tmp_lmdb), map_size=256 * 1024 ** 3, subdir=True, lock=True)
        self.overwrite = overwrite
        self.keys = {split: [] for split in ('train', 'val', 'test')}
        self.rows = {split: [] for split in self.keys}
        self.channel_montages = Counter()
        self.source_rates = Counter()

    def add(self, candidate: Candidate, split: str, sample: np.ndarray, channels: list[str], channel_indices: list[int], source_rate: float, label_id: int) -> None:
        if sample.shape != (len(channels), WINDOW_SAMPLES):
            raise ValueError(f'unexpected sample shape {sample.shape}')
        sample = np.ascontiguousarray(sample, dtype=np.float32).reshape(len(channels), PATCH_COUNT, PATCH_SIZE)
        start_ms = round(candidate.start_seconds * 1000)
        key_text = f'{safe_id(candidate.dataset)}_{safe_id(candidate.subject)}_{safe_id(candidate.session)}_{safe_id(candidate.label)}_{start_ms:010d}'
        key = key_text.encode('utf-8')
        payload = {'sample': sample, 'label': candidate.label, 'raw_label': label_id, 'subject': candidate.subject, 'session': candidate.session, 'source_id': candidate.path.name}
        with self.env.begin(write=True) as txn:
            if not txn.put(key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), overwrite=False):
                raise RuntimeError(f'duplicate LMDB key: {key_text}')
        self.keys[split].append(key)
        self.rows[split].append({'canonical_channel_indices': channel_indices, 'channels': channels, 'dataset': candidate.dataset, 'id': key_text, 'input_scale': 1.0, 'input_shape': [len(channels), WINDOW_SAMPLES], 'label': candidate.label, 'lmdb_key': key_text, 'lmdb_path': str(self.final_lmdb), 'patch_count': PATCH_COUNT, 'patch_size': PATCH_SIZE, 'raw_label': label_id, 'sampling_rate': TARGET_RATE, 'session': candidate.session, 'skip_trial_normalization': False, 'source_file': str(candidate.path), 'source_sampling_rate': source_rate, 'source_start_seconds': candidate.start_seconds, 'subject': candidate.subject, 'task': TASKS[candidate.dataset]})
        self.channel_montages[tuple(channels)] += 1
        self.source_rates[str(source_rate)] += 1

    def finish(self, assignments: dict[str, str], labels: dict[str, int], errors: list[dict], protocol: dict) -> dict:
        with self.env.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(self.keys, protocol=pickle.HIGHEST_PROTOCOL))
        self.env.sync()
        self.env.close()
        for (split, rows) in self.rows.items():
            with (self.tmp_manifests / f'{split}.jsonl').open('w', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + '\n')
        split_summary = {}
        for (split, rows) in self.rows.items():
            participants = sorted({row['subject'] for row in rows})
            split_summary[split] = {'participants': participants, 'participant_count': len(participants), 'samples': len(rows), 'label_distribution': dict(sorted(Counter((row['label'] for row in rows)).items()))}
        subject_sets = {split: set(value['participants']) for (split, value) in split_summary.items()}
        overlap = {'train_val': sorted(subject_sets['train'] & subject_sets['val']), 'train_test': sorted(subject_sets['train'] & subject_sets['test']), 'val_test': sorted(subject_sets['val'] & subject_sets['test'])}
        if any(overlap.values()):
            raise RuntimeError(f'subject leakage detected: {overlap}')
        result = {'dataset': self.dataset, 'status': 'complete', 'source': str(DATASET_ROOTS[self.dataset]), 'participants': len(assignments), 'sampling_rate': TARGET_RATE, 'window_seconds': WINDOW_SECONDS, 'patch_size': PATCH_SIZE, 'patch_count': PATCH_COUNT, 'labels': labels, 'source_sampling_rates': dict(sorted(self.source_rates.items())), 'splits': split_summary, 'subject_overlap': overlap, 'failed_records': errors, 'channel_montages': [{'channels': list(channels), 'sample_count': count} for (channels, count) in sorted(self.channel_montages.items(), key=lambda item: (-item[1], item[0]))], **protocol}
        text = json.dumps(result, indent=2, ensure_ascii=True) + '\n'
        (self.tmp_manifests / 'protocol.json').write_text(text)
        (self.tmp_lmdb / 'protocol.json').write_text(text)
        if self.final_lmdb.exists():
            if not self.overwrite:
                raise FileExistsError(self.final_lmdb)
            shutil.rmtree(self.final_lmdb)
        if self.final_manifests.exists():
            if not self.overwrite:
                raise FileExistsError(self.final_manifests)
            shutil.rmtree(self.final_manifests)
        self.final_lmdb.parent.mkdir(parents=True, exist_ok=True)
        self.final_manifests.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_lmdb.rename(self.final_lmdb)
        self.tmp_manifests.rename(self.final_manifests)
        return result

def preprocess_chb(path: Path) -> tuple[np.ndarray, list[str], list[int], float]:
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    channels = list(CHB_CHANNEL_MAP.values())
    data = np.stack([payload[name] for name in CHB_CHANNEL_MAP], axis=0).astype(np.float32, copy=False)
    source_rate = 256.0
    sos = butter(4, [0.3, 75.0], btype='bandpass', fs=source_rate, output='sos')
    data = sosfiltfilt(sos, data, axis=-1)
    (b_notch, a_notch) = iirnotch(60.0, 30.0, fs=source_rate)
    data = filtfilt(b_notch, a_notch, data, axis=-1)
    data = resample_poly(data, 25, 32, axis=-1).astype(np.float32, copy=False)
    return (data, channels, [CANONICAL_INDEX[name] for name in channels], source_rate)

def preprocess_event_edf(dataset: str, path: Path) -> tuple[np.ndarray, list[str], list[int], float]:
    if dataset != 'HMC_SLEEP':
        from prepare_new_scalp_native200 import SourceRecord
        record = SourceRecord(dataset, TASKS[dataset], '', '', '', path)
        return preprocess_raw(record)
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    source_rate = float(raw.info['sfreq'])
    (raw, channels, channel_indices) = select_canonical_eeg(raw, minimum_channels=4)
    raw.filter(l_freq=0.3, h_freq=75.0, method='fir', n_jobs=1, verbose='ERROR')
    raw.notch_filter(freqs=[50.0], method='fir', n_jobs=1, verbose='ERROR')
    if round(source_rate) != TARGET_RATE:
        raw.resample(TARGET_RATE, npad='auto', n_jobs=1, verbose='ERROR')
    data = raw.get_data().astype(np.float32, copy=False) * np.float32(1000000.0)
    del raw
    return (data, channels, channel_indices, source_rate)

def validate_dataset(dataset: str, output_root: Path, manifest_root: Path) -> None:
    protocol = json.loads((manifest_root / dataset / 'protocol.json').read_text())
    assert protocol['status'] == 'complete'
    assert not any(protocol['subject_overlap'].values())
    env = lmdb.open(str(output_root / dataset / 'processed_average'), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            keys = pickle.loads(txn.get(b'__keys__'))
            for split in ('train', 'val', 'test'):
                assert len(keys[split]) == protocol['splits'][split]['samples']
                for key in keys[split][:min(3, len(keys[split]))]:
                    sample = pickle.loads(txn.get(key))['sample']
                    assert sample.dtype == np.float32
                    assert sample.ndim == 3 and sample.shape[1:] == (PATCH_COUNT, PATCH_SIZE)
                    assert np.isfinite(sample).all()
    finally:
        env.close()
    print(f'[{dataset}] validation passed', flush=True)

def build_dataset(dataset: str, output_root: Path, manifest_root: Path, cap: int, seed: int, overwrite: bool, max_subjects_per_group: int | None) -> dict:
    complete_marker = manifest_root / dataset / 'protocol.json'
    if complete_marker.is_file() and (not overwrite):
        current = json.loads(complete_marker.read_text())
        if current.get('status') == 'complete':
            print(f'[{dataset}] already complete; skipping', flush=True)
            return current
    print(f'[{dataset}] discovering event windows', flush=True)
    selected = capped_candidates(DISCOVERERS[dataset](DATASET_ROOTS[dataset]), cap, seed)
    if max_subjects_per_group is not None:
        subjects_by_group: dict[str, set[str]] = defaultdict(set)
        for candidate in selected:
            subjects_by_group[candidate.split_group].add(candidate.subject)
        retained = {subject for subjects in subjects_by_group.values() for subject in sorted(subjects)[:max_subjects_per_group]}
        selected = [candidate for candidate in selected if candidate.subject in retained]
    if not selected:
        raise RuntimeError(f'{dataset}: no event windows discovered')
    assignments = assign_splits(dataset, selected, seed)
    labels = {label: index for (index, label) in enumerate(sorted({item.label for item in selected}))}
    grouped: dict[Path, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        grouped[candidate.path].append(candidate)
    print(f'[{dataset}] selected {len(selected)} windows from {len(grouped)} files and {len(assignments)} subjects', flush=True)
    writer = NativeWriter(dataset, output_root, manifest_root, overwrite)
    errors = []
    for (file_index, (path, candidates)) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0])), start=1):
        try:
            if dataset == 'CHB_MIT':
                (data, channels, channel_indices, source_rate) = preprocess_chb(path)
            else:
                (data, channels, channel_indices, source_rate) = preprocess_event_edf(dataset, path)
            for candidate in candidates:
                start = round(candidate.start_seconds * TARGET_RATE)
                sample = data[:, start:start + WINDOW_SAMPLES]
                if sample.shape[-1] != WINDOW_SAMPLES:
                    raise ValueError(f'window {candidate.start_seconds:.3f}s is outside resampled recording {data.shape[-1]}')
                writer.add(candidate, assignments[candidate.subject], sample, channels, channel_indices, source_rate, labels[candidate.label])
            print(f'[{dataset}] {file_index}/{len(grouped)} {path.name}: {len(candidates)} windows', flush=True)
        except Exception as exc:
            errors.append({'path': str(path), 'error': repr(exc), 'candidate_windows': len(candidates)})
            print(f'[{dataset}] ERROR {path}: {exc!r}', file=sys.stderr, flush=True)
    result = writer.finish(assignments, labels, errors, {'max_windows_per_subject_label': cap, 'preprocessing': {'channel_selection': 'native scalp channels mapped to the canonical 75-channel montage', 'resample_hz': TARGET_RATE, 'bandpass_hz': [0.3, 75.0], 'notch_hz': 50.0 if dataset == 'HMC_SLEEP' else 60.0, 'stored_unit': 'uV', 'window_normalization': 'deferred to the training loader', 'event_sampling': 'deterministic hash reservoir per subject and label', 'tuev_duplicate_policy': 'duplicate channel annotations with equal start/stop/label are removed', 'tusz_label_policy': 'bckg is background; every seizure subtype is seizure'}, 'split_policy': {'TUEV': 'official eval is test; official train subjects are split 80/20 train/val', 'TUSZ': 'official train/dev/eval are train/val/test', 'HMC_SLEEP': 'deterministic subject-disjoint 70/15/15 split', 'CHB_MIT': 'chb01-chb20 train, chb21-chb22 val, chb23-chb24 test'}[dataset], 'seed': seed})
    print(f'[{dataset}] complete: ' + ', '.join((f"{split}={value['samples']}" for (split, value) in result['splits'].items())) + f', errors={len(errors)}', flush=True)
    validate_dataset(dataset, output_root, manifest_root)
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=sorted(DATASET_ROOTS), default=sorted(DATASET_ROOTS))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifest-root', type=Path, required=True)
    parser.add_argument('--max-windows-per-subject-label', type=int, default=40)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max-subjects-per-group', type=int)
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
        build_dataset(dataset, args.output_root, args.manifest_root, args.max_windows_per_subject_label, args.seed, args.overwrite, args.max_subjects_per_group)
if __name__ == '__main__':
    main()

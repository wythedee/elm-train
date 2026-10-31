#!/usr/bin/env python3
"""Recover one supplied .zst processed file into a NEW server-side path.

Never changes archives/symlinks or deletes old data. Partial output is retained
after a failure. A bounded amount of disk and time is required explicitly.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def stamp(pid):
    p = Path('/proc')/str(pid)
    try:
        stat = (p/'stat').read_text(); rest = stat[stat.rfind(')')+2:].split()
        return (int(rest[19]), (p/'cmdline').read_bytes())
    except OSError:
        return None


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--max-bytes', type=int, required=True)
    p.add_argument('--reserve-gib', type=int, default=64)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    if sys.platform != 'linux':
        p.error('Data recovery stays on the authorized Linux server')
    if not args.source.is_file() or args.source.suffix != '.zst':
        p.error('Supply an existing .zst archive')
    args.destination.parent.resolve(strict=True)
    partial = args.destination.with_suffix(args.destination.suffix+'.partial')
    if args.destination.exists() or partial.exists():
        p.error('Refusing to overwrite an existing or partial recovery')
    if shutil.disk_usage(args.destination.parent).free < args.max_bytes+args.reserve_gib*1024**3:
        p.error('Not enough capacity for the declared maximum plus reserve')
    executable = shutil.which('zstd')
    if not executable:
        p.error('Existing zstd executable required; no installation is performed')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Recovery interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    sidecar = args.destination.with_suffix(args.destination.suffix+'.recovery.json')
    with args.destination.with_suffix(args.destination.suffix+'.recover.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        command = [executable, '-dc', '--', str(args.source)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        identity = None
        for _ in range(100):
            current = stamp(process.pid)
            if current and current[1].split(b'\0')[:-1] == [s.encode() for s in command]:
                identity = current; break
            time.sleep(.01)
        if identity is None:
            raise RuntimeError('Decompressor exact identity unresolved')
        total, digest = 0, hashlib.sha256()
        try:
            with partial.open('xb') as stream:
                while True:
                    if time.time() > args.deadline:
                        raise TimeoutError('Original recovery deadline expired')
                    if shutil.disk_usage(args.destination.parent).free < args.reserve_gib*1024**3:
                        raise OSError('Recovery disk reserve')
                    chunk = process.stdout.read(64*1024**2)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > args.max_bytes:
                        raise OSError('Declared output bound exceeded')
                    stream.write(chunk); digest.update(chunk)
                    write(sidecar, {'status':'running','written_bytes':total,'deadline':args.deadline,
                                    'pid':process.pid,'time':time.time()})
                stream.flush(); os.fsync(stream.fileno())
            if process.wait() != 0:
                raise RuntimeError('zstd integrity/decompression failure')
            partial.replace(args.destination)
            result = {'status':'complete','bytes':total,'sha256':digest.hexdigest(),
                      'source_archive_unchanged':True,'old_paths_unchanged':True,
                      'time':time.time(),'deadline':args.deadline}
            write(sidecar,result);print(json.dumps(result),flush=True)
        except BaseException:
            if process.poll() is None:
                if stamp(process.pid) != identity:
                    raise RuntimeError('Decompressor identity changed; no signal sent')
                process.terminate(); process.wait()
            raise
        finally:
            process.stdout.close()


if __name__=='__main__':
    main()

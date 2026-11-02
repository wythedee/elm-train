"""Portable program-scored Oracle tasks; wording comes from the frozen 27B output."""
from __future__ import annotations
import copy
import json
import random
import re
from pathlib import Path
from . import tasks
from .io import read

OPERATIONS = ("read_level", "compare_level", "choose_higher", "choose_lower")
OPTIONS = {"read_level": tasks.LEVELS, "compare_level": ["lower", "equal", "higher"],
           "choose_higher": ["A", "B", "tie"], "choose_lower": ["A", "B", "tie"]}
SYSTEM = ("Read the supplied EEG statistical facts. Each value is an ordinal bin, ordered low < mid < high. "
          "Compare bins, not continuous values. 'equal' or 'tie' means the same bin. "
          "Answer using exactly one JSON object with key answer, without explanation.")


def gold(operation, levels):
    a = tasks.LEVELS.index(levels[0])
    if operation == "read_level":
        return {"answer": levels[0]}
    b = tasks.LEVELS.index(levels[1])
    if operation == "compare_level":
        return {"answer": "lower" if a < b else "higher" if a > b else "equal"}
    return {"answer": "tie" if a == b else ("A" if (a > b) == (operation == "choose_higher") else "B")}


def pool(records, split, limit, seed):
    wording = read(Path(__file__).resolve().parents[1] / "metadata/oracle_wordings.json")["templates"]
    grouped = {op: [r for r in wording if r["operation"] == op and r["partition"] == split]
               for op in OPERATIONS}
    groups = {k: [] for k in ["label"] + ["feature:" + f for f in tasks.FIELDS] + ["report"] + list(OPERATIONS)}
    for i, record in enumerate(records):
        for task in list(groups)[:8]:
            e = tasks.example(record, task);e["records"] = [record];groups[task].append(e)
        for j, op in enumerate(OPERATIONS):
            template = grouped[op][i % len(grouped[op])]
            other = records[(i + 1 + j) % len(records)]
            rr = [record] if op == "read_level" else [record, other]
            field = tasks.FIELDS[(i + j) % 6]
            question = template["wording"].format(field=field, record_a="A", record_b="B")
            e = {"id": record["id"] + ":" + op + ":" + field, "task": op,
                 "question": question, "field": field, "template_id": template["id"],
                 "records": rr, "record": record,
                 "expected": gold(op, [r["bins"][field] for r in rr])}
            groups[op].append(e)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    result = []
    # Original declared 20% basic / 80% simple-operation mixture, fixed before outputs.
    basic = list(groups)[:8]
    for i in range(limit):
        keys = basic if i % 5 == 0 else list(OPERATIONS)
        offset = i // 5 if i % 5 == 0 else i - (i // 5 + 1)
        task = keys[offset % len(keys)]
        if not groups[task]:
            raise ValueError("Not enough unique queries for declared software pool: " + task)
        result.append(groups[task].pop())
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate Oracle query IDs")
    return result


def messages(item, teacher=False):
    if item["task"] not in OPERATIONS:
        return tasks.messages(item, teacher)
    body = "EEG statistical facts:\n"
    for i, record in enumerate(item["records"]):
        facts = tasks.dumps({"label": record["label"], "statistics": record["bins"]}) if teacher else "__SIMPLE_ORACLE_%d__" % i
        body += "Record " + chr(65+i) + ":\n" + facts + "\n"
    body += "Question: " + item["question"] + "\nAllowed answer strings: " + json.dumps(OPTIONS[item["task"]])
    body += '\nOutput schema: {"answer":"<one allowed string>"}'
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def render(tokenizer, embedding, item, prefixes=None, teacher=False):
    import torch
    if item["task"] not in OPERATIONS:
        return tasks.render(tokenizer, embedding, item, None if teacher else prefixes[0], teacher)
    text = tokenizer.apply_chat_template(messages(item, teacher), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    if teacher:
        return embedding(torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], device=embedding.weight.device)).detach()
    if len(prefixes) != len(item["records"]):
        raise ValueError("Wrong number of per-record prefixes")
    pieces = []
    for chunk in re.split(r"(__SIMPLE_ORACLE_\d+__)", text):
        m = re.fullmatch(r"__SIMPLE_ORACLE_(\d+)__", chunk)
        if m:
            p = prefixes[int(m[1])]
            if p.shape != (1, 80, 2560):raise ValueError("Oracle prefix shape changed")
            pieces.append(p[0].to(device=embedding.weight.device, dtype=embedding.weight.dtype))
        elif chunk:
            pieces.append(embedding(torch.tensor(tokenizer(chunk, add_special_tokens=False)["input_ids"], device=embedding.weight.device)).detach())
    return torch.cat(pieces)


def score(text, item):
    if item["task"] not in OPERATIONS:
        return tasks.score(text, item)
    try:value = json.loads(text.strip())
    except (ValueError, TypeError):value = None
    valid = isinstance(value, dict) and set(value) == {"answer"} and value["answer"] in OPTIONS[item["task"]]
    return {"prediction": value, "valid": bool(valid), "correct": bool(valid and value == item["expected"]),
            "components": {"answer": bool(valid and value == item["expected"])}}


def leak_check(item):
    changed = copy.deepcopy(item);changed["expected"] = {"changed": True}
    for record in changed["records"]:
        record["label"] = "changed";record["bins"] = {f: "high" for f in tasks.FIELDS}
    if messages(changed) != messages(item):raise ValueError("Gold metadata leaked into student text")

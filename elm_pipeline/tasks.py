"""Current class/statistic/report protocol, without private task-module imports."""
from __future__ import annotations

import copy
import json
import re

FIELDS = ["alpha_relative_power", "theta_beta_ratio", "spectral_entropy",
          "hjorth_mobility", "hjorth_complexity", "line_length"]
LEVELS = ["low", "mid", "high"]
QUESTION = 'Read the EEG evidence and return its observed class as {"label":"CLASS"}. Output only JSON.'
REPORT_QUESTION = ('Write a factual machine-readable EEG report. Return only JSON with keys "label" and "statistics". '
                   '"statistics" must contain exactly these six fields: alpha_relative_power, theta_beta_ratio, '
                   'spectral_entropy, hjorth_mobility, hjorth_complexity, line_length. '
                   'Each statistic is low, mid or high. State the encoded facts, without diagnoses or extra fields.')
SYSTEM = "Read the supplied EEG observed facts. Return only the requested JSON, without extra fields or diagnoses."


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def example(record, task="label", question=None):
    if task == "label":
        expected, wording = {"label": record["label"]}, QUESTION
    elif task == "report":
        expected = {"label": record["label"],
                    "statistics": {f: record["bins"][f] for f in FIELDS}}
        wording = REPORT_QUESTION
    elif task.startswith("feature:") and task.split(":", 1)[1] in FIELDS:
        field = task.split(":", 1)[1]
        expected = {field: record["bins"][field]}
        wording = ('Read the EEG evidence. What is the ' + field
                   + ' level? Return only a JSON object with key "' + field
                   + '" and a value low, mid or high.')
    else:
        raise ValueError("Unsupported task: " + task)
    return {"id": record["id"] + ":" + task, "task": task,
            "question": question if question is not None else wording,
            "expected": expected, "record": record}


def messages(item, teacher=False):
    r = item["record"]
    facts = dumps({"label": r["label"], "statistics": {f: r["bins"][f] for f in FIELDS}}) if teacher else "__SIMPLE_ORACLE_0__"
    if "__SIMPLE_ORACLE_" in item["question"]:
        raise ValueError("Question contains a reserved input marker")
    body = "EEG statistical facts:\nRecord A:\n" + facts + "\nQuestion: " + item["question"]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def render(tokenizer, embedding, item, prefix=None, teacher=False):
    import torch
    text = tokenizer.apply_chat_template(messages(item, teacher), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    device = embedding.weight.device
    if teacher:
        if prefix is not None:
            raise ValueError("Text teacher must not receive an EEG prefix")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return embedding(torch.tensor(ids, device=device)).detach()
    if prefix is None or prefix.shape != (1, 80, 2560):
        raise ValueError("Student needs exactly one 80×2560 prefix")
    if text.count("__SIMPLE_ORACLE_0__") != 1:
        raise ValueError("Chat template altered the prefix marker")
    parts = []
    for chunk in re.split(r"(__SIMPLE_ORACLE_0__)", text):
        if chunk == "__SIMPLE_ORACLE_0__":
            parts.append(prefix[0].to(device=device, dtype=embedding.weight.dtype))
        elif chunk:
            ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
            parts.append(embedding(torch.tensor(ids, device=device)).detach())
    return torch.cat(parts)


def score(text, item):
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError):
        value = None
    expected, task = item["expected"], item["task"]
    valid = isinstance(value, dict)
    if task == "report":
        stats = value.get("statistics") if isinstance(value, dict) else None
        valid = (valid and set(value) == {"label", "statistics"}
                 and isinstance(value.get("label"), str)
                 and isinstance(stats, dict) and set(stats) == set(FIELDS)
                 and all(stats[f] in LEVELS for f in FIELDS))
        components = {"label": isinstance(value, dict) and value.get("label") == expected["label"]}
        components.update({f: isinstance(stats, dict) and stats.get(f) == expected["statistics"][f]
                           for f in FIELDS})
    else:
        valid = valid and set(value) == set(expected)
        if task == "label":
            valid = valid and isinstance(value.get("label"), str)
        else:
            field = task.split(":", 1)[1]
            valid = valid and value.get(field) in LEVELS
        components = {k: isinstance(value, dict) and value.get(k) == v for k, v in expected.items()}
    return {"prediction": value, "valid": bool(valid),
            "correct": bool(valid and value == expected), "components": components}


def no_leak_fixture():
    r = {"id": "fixture", "dataset": "fixture", "label": "left_hand",
         "bins": {f: "low" for f in FIELDS}}
    for task in ["label", "report"] + ["feature:" + f for f in FIELDS]:
        e = example(r, task)
        altered = copy.deepcopy(e)
        altered["expected"] = {"irrelevant": "changed"}
        altered["record"]["label"] = "right_hand"
        altered["record"]["bins"] = {f: "high" for f in FIELDS}
        assert messages(e) == messages(altered)
        assert score(dumps(e["expected"]), e)["correct"]
    wrong = example(r, "report")
    answer = copy.deepcopy(wrong["expected"])
    answer["label"] = "EEG Report"
    result = score(dumps(answer), wrong)
    assert result["valid"] and not result["correct"]
    return {"status": "passed", "gold_metadata_not_student_text": True,
            "valid_but_wrong_report_rejected_as_correct": True}

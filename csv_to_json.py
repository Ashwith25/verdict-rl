import re
from typing import List, Dict, Tuple
import json
from trajectory import trajectory_A

BRACKET_FLOATS_RE = re.compile(r"\[([^\]]*)\]")

def parse_vector(bracket_text: str) -> List[float]:
    parts = [p.strip() for p in bracket_text.split(",") if p.strip()]
    return [float(p) for p in parts]

def parse_line(line: str) -> Tuple[List[float], List[float], List[float]]:
    matches = BRACKET_FLOATS_RE.findall(line)
    if len(matches) != 3:
        raise ValueError(f"Expected 3 bracketed vectors, got {len(matches)}: {line}")
    return (
        parse_vector(matches[0]),
        parse_vector(matches[1]),
        parse_vector(matches[2]),
    )

def trajectory_string_to_json(input_str: str) -> List[Dict]:
    records = []
    state_id = 0

    lines = [l.strip() for l in input_str.strip().splitlines() if l.strip()]

    for t, line in enumerate(lines):
        state, action, next_state = parse_line(line)

        record = {
            "t": t,
            "state_ID": state_id,
            "next_state_ID": state_id + 1
        }

        for i, val in enumerate(state):
            record[f"state_{i}"] = val

        for i, val in enumerate(action):
            record[f"action_{i}"] = val

        # Optional: include full next state values
        # for i, val in enumerate(next_state):
        #     record[f"next_state_{i}"] = val

        records.append(record)
        state_id += 1

    return records
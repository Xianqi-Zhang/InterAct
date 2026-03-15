import os
import re
import shutil
from pathlib import Path

import numpy as np


def _tokenize_for_text2interaction(text: str) -> str:
    # text2interaction expects "word/POS" tokens joined by spaces.
    words = re.findall(r"[A-Za-z]+", text.lower())
    return " ".join(f"{w}/OTHER" for w in words)


def _parse_description_line(line: str):
    # Example: "3-57 grasp, both hands"
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s+(.*)$", line.strip())
    if m is None:
        return None
    start = int(m.group(1))
    end = int(m.group(2))
    action_part = m.group(3).strip()
    if end <= start:
        return None
    return start, end, action_part


def _build_caption(action_part: str, obj_name: str) -> str:
    # "grasp, both hands" -> "A person grasp the box with both hands."
    parts = [x.strip() for x in action_part.split(",", 1)]
    verb = parts[0] if parts else action_part
    hand = parts[1] if len(parts) > 1 else ""
    if hand:
        return f"A person {verb} the {obj_name} with {hand}."
    return f"A person {verb} the {obj_name}."


def main():
    data_root = Path("./data")
    dataset = "arctic"
    dataset_path = data_root / dataset
    sequence_root = dataset_path / "sequences"
    sequence_seg_root = dataset_path / "sequences_seg"
    description_root = dataset_path / "description"
    sequence_seg_root.mkdir(parents=True, exist_ok=True)

    if not sequence_root.exists():
        raise FileNotFoundError(f"Missing sequence root: {sequence_root}")
    if not description_root.exists():
        raise FileNotFoundError(f"Missing description root: {description_root}")

    for desc_file in description_root.rglob("description.txt"):
        rel = desc_file.relative_to(description_root)
        # <subject>/<seq>/description.txt -> <subject>_<seq>
        if len(rel.parts) < 3:
            continue
        subject = rel.parts[0]
        seq = rel.parts[1]
        seq_name = f"{subject}_{seq}"
        seq_path = sequence_root / seq_name
        human_path = seq_path / "human.npz"
        object_path = seq_path / "object.npz"
        if not human_path.exists() or not object_path.exists():
            continue

        human = np.load(human_path, allow_pickle=True)
        obj = np.load(object_path, allow_pickle=True)
        t = int(human["trans"].shape[0])
        obj_name = str(obj["name"]).lower()

        lines = [x.strip() for x in desc_file.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        for line in lines:
            parsed = _parse_description_line(line)
            if parsed is None:
                continue
            start, end, action_part = parsed
            if start >= t:
                continue
            end = min(end, t)
            if end - start < 2:
                continue

            seg_name = f"{seq_name}_{start}"
            seg_path = sequence_seg_root / seg_name
            seg_path.mkdir(parents=True, exist_ok=True)

            human_seg = {}
            for k in human.files:
                if k in {"betas", "beta", "gender", "vtemp"}:
                    human_seg[k] = human[k]
                else:
                    human_seg[k] = human[k][start:end]
            np.savez(seg_path / "human.npz", **human_seg)

            obj_seg = {}
            for k in obj.files:
                if k == "name":
                    obj_seg[k] = obj[k]
                else:
                    obj_seg[k] = obj[k][start:end]
            np.savez(seg_path / "object.npz", **obj_seg)

            caption = _build_caption(action_part, obj_name)
            tokens = _tokenize_for_text2interaction(caption)
            (seg_path / "text.txt").write_text(f"{caption}#{tokens}#0.0#0.0\n", encoding="utf-8")
            # Keep compatibility with other segmented datasets.
            shutil.copy(seg_path / "text.txt", seg_path / "action.txt")

    print(f"[process_text_arctic] done: {sequence_seg_root}")


if __name__ == "__main__":
    main()

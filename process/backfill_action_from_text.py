import argparse
from pathlib import Path


def backfill_dataset(dataset_root: Path, dataset: str, dry_run: bool = False) -> tuple[int, int]:
    seq_root = dataset_root / dataset / "sequences_canonical"
    if not seq_root.is_dir():
        print(f"[WARN] Skip dataset={dataset}: missing {seq_root}")
        return 0, 0

    filled = 0
    total = 0
    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        total += 1
        text_path = seq_dir / "text.txt"
        action_path = seq_dir / "action.txt"
        if action_path.exists() or not text_path.exists():
            continue
        if not dry_run:
            action_path.write_text(text_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        filled += 1
    return filled, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing action.txt from text.txt.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="chairs,neuraldome,imhd,omomo",
        help="Comma-separated dataset names to process.",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing files.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    total_fill = 0
    total_seq = 0
    for dataset in datasets:
        filled, seqs = backfill_dataset(data_root, dataset, dry_run=args.dry_run)
        total_fill += filled
        total_seq += seqs
        print(f"[{dataset}] filled={filled} total_sequences={seqs}")

    print(f"[done] filled={total_fill} scanned_sequences={total_seq} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()

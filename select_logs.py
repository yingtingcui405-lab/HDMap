#!/usr/bin/env python3
"""Index TbV logs and select scene-level map-change candidates. Python 3.9+.

Reads annotations and paths only. Never edits, copies, or deletes sensor data.
This is dataset curation using ground truth, not a change-detection model.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYERS = ("laneSegments", "pedCrossings", "drivableAreas")
CAMERAS = ("ring_front_center", "ring_front_left", "ring_front_right",
           "ring_rear_left", "ring_rear_right", "ring_side_left", "ring_side_right")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_lookup(splits):
    result = {}
    for split, values in splits.items():
        ids = values.values() if isinstance(values, dict) else values
        for log_id in ids:
            if log_id in result:
                raise ValueError(f"Duplicate split membership: {log_id}")
            result[log_id] = split
    return result


def index_logs(root):
    if not root.is_dir():
        raise ValueError(f"TbV root does not exist: {root}")
    shards = sorted(p for p in root.glob("TbV_v1.0_shard*") if p.is_dir())
    result = {}
    for parent in shards or [root]:
        for path in sorted(parent.iterdir()):
            if path.name.startswith(".") or not path.is_dir():
                continue
            # Support both the downloaded shard layout and a flattened log root.
            if "__" not in path.name:
                continue
            if path.name in result:
                raise ValueError(f"Duplicate log_id: {path.name}: {result[path.name]} and {path}")
            result[path.name] = path
    if not result:
        raise ValueError(f"No log directories found in {root}")
    return result


def parse_changes(data, log_id):
    if data.get("filename") != log_id:
        raise ValueError(f"Annotation filename mismatch: {log_id}")
    changed = []
    for layer in LAYERS:
        elements = data.get(layer)
        if not isinstance(elements, dict):
            raise ValueError(f"{log_id}: missing/invalid layer {layer}")
        for element_id, element in elements.items():
            codes = element.get("changes")
            if (not isinstance(codes, list) or not codes
                    or any(type(c) is not int or c < 0 for c in codes)
                    or (0 in codes and codes != [0])):
                raise ValueError(f"{log_id}/{element_id}: invalid changes={codes!r}")
            if any(c != 0 for c in codes):
                # Preserve all raw fields and codes. No guessing old/new time direction.
                changed.append({"log_id": log_id, "layer": layer,
                                "element_id": element_id, "annotation": element})
    return changed


def classify(tbv_split, at_split, event_count, changed_count):
    if tbv_split == "val" and at_split == "test":
        if event_count is None or changed_count is None:
            return "review"
        if event_count > 0 or changed_count > 0:
            return "real_change"
        return "real_no_change_control"
    if tbv_split == "train" and at_split in ("train", "val"):
        return "constructed_pool"
    if tbv_split == "test" and at_split is None:
        return "unlabeled"
    return "review"


def inspect_layout(path):
    files = ["city_SE3_egovehicle.feather", "calibration/intrinsics.feather",
             "calibration/egovehicle_SE3_sensor.feather"]
    missing = [name for name in files if not (path / name).is_file()]
    map_files = sorted((path / "map").glob("log_map_archive_*.json"))
    if len(map_files) != 1:
        missing.append(f"expected_one_vector_map_found_{len(map_files)}")
    dirs = ["sensors/lidar"] + [f"sensors/cameras/{c}" for c in CAMERAS]
    missing.extend(name for name in dirs if not (path / name).is_dir())
    return {"layout_missing": missing, "vector_map_paths": [str(p) for p in map_files],
            "layout_check": "paths_only_not_sensor_integrity"}


def run(args):
    tbv_root = args.tbv_root.expanduser().resolve()
    at_root = args.argotweak_root.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise ValueError(f"Output exists; choose a NEW --out directory: {out}")
    if any(out == p or p in out.parents for p in (tbv_root, at_root)):
        raise ValueError("Output must be outside the two raw dataset directories")

    tbv = split_lookup(read_json(metadata / "tbv_splits.json"))
    at = split_lookup(read_json(metadata / "argotweak_splits.json"))
    annotations = read_json(metadata / "tbv_val_split_annotations.json")
    events_by_log = {}
    for item in annotations:
        log_id = item["log_id"]
        if log_id in events_by_log or not isinstance(item.get("events"), list):
            raise ValueError(f"Invalid/duplicate TbV event record: {log_id}")
        events_by_log[log_id] = item
    if set(events_by_log) != {k for k, v in tbv.items() if v == "val"}:
        raise ValueError("TbV public annotations do not exactly match its val split")

    logs = index_logs(tbv_root)
    at_paths = {}
    for folder in ("train+val", "test"):
        directory = at_root / folder
        if not directory.is_dir():
            raise ValueError(f"Missing ArgoTweak directory: {directory}")
        for path in sorted(directory.glob("*.json")):
            if path.name.startswith("."):
                continue
            expected = "test" if at.get(path.stem) == "test" else "train+val"
            if path.stem not in at or expected != folder or path.stem in at_paths:
                raise ValueError(f"Unexpected split/duplicate annotation: {path}")
            at_paths[path.stem] = path

    rows, changed_elements, event_rows = [], [], []
    for number, log_id in enumerate(sorted(logs), 1):
        path = logs[log_id]
        ts, ats = tbv.get(log_id), at.get(log_id)
        event_record = events_by_log.get(log_id)
        events = event_record["events"] if event_record is not None else None
        at_path = at_paths.get(log_id)
        changes = None
        problems = []
        should_parse = ats == "test" or (args.scan_constructed and ats in ("train", "val"))
        if ats is not None and at_path is None:
            problems.append("missing_argotweak_annotation")
        if should_parse and at_path:
            try:
                changes = parse_changes(read_json(at_path), log_id)
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
                problems.append(f"annotation_parse_error: {exc}")
        n_events = len(events) if events is not None else None
        n_changed = len(changes) if changes is not None else None
        bucket = classify(ts, ats, n_events, n_changed)
        if ts == "val" and n_events is not None and n_changed is not None:
            if bool(n_events) != bool(n_changed):
                problems.append("tbv_argotweak_scene_label_disagreement")
        evidence = []
        if n_events:
            evidence.append("tbv_public_spatial_events")
        if n_changed:
            evidence.append("argotweak_nonzero_changes")
        row = {
            "log_id": log_id, "log_dir": str(path), "shard": path.parent.name,
            "tbv_split": ts, "argotweak_split": ats, "bucket": bucket,
            "argotweak_path": str(at_path) if at_path else None,
            "argotweak_scanned": changes is not None,
            "tbv_event_count": n_events, "argotweak_changed_element_count": n_changed,
            "tbv_event_types": sorted({e["change_type"] for e in events}) if events else [],
            "argotweak_change_codes": sorted({c for e in changes for c in e["annotation"]["changes"]}) if changes is not None else None,
            "changed_elements_by_layer": dict(Counter(e["layer"] for e in changes)) if changes is not None else None,
            "selection_evidence": evidence, "review_reasons": problems,
            "temporal_extent": "not_selected_yet", "permanence": "not_inferred",
        }
        if bucket.startswith("real_") or ts == "val":
            row.update(inspect_layout(path))
            if row["layout_missing"]:
                problems.append("missing_required_paths")
        rows.append(row)
        if events is not None:
            for index, event in enumerate(events):
                event_rows.append({"log_id": log_id, "event_index": index,
                                   "source": "tbv_val_split_annotations.json", "event": event})
        for change in changes or []:
            change["argotweak_split"] = ats
            change["annotation_path"] = str(at_path)
            changed_elements.append(change)
        if number % 50 == 0 or number == len(logs):
            print(f"Indexed {number}/{len(logs)} logs", flush=True)

    buckets = dict(Counter(r["bucket"] for r in rows))
    files = {
        "all_logs.jsonl": rows,
        "real_change_logs.jsonl": [r for r in rows if r["bucket"] == "real_change"],
        "real_no_change_controls.jsonl": [r for r in rows if r["bucket"] == "real_no_change_control"],
        "constructed_pool.jsonl": [r for r in rows if r["bucket"] == "constructed_pool"],
        "unlabeled_logs.jsonl": [r for r in rows if r["bucket"] == "unlabeled"],
        "review_logs.jsonl": [r for r in rows if r["review_reasons"] or r["bucket"] == "review"],
        "tbv_events.jsonl": event_rows,
        "argotweak_changed_elements.jsonl": changed_elements,
    }
    # Missing inputs are explicitly reported, never silently treated as no-change.
    missing_logs = sorted(set(tbv) - set(logs))
    missing_at = sorted(set(at) - set(at_paths))
    manifest = read_json(metadata / "sources.json")
    input_files = [metadata / n for n in ("tbv_splits.json", "argotweak_splits.json", "tbv_val_split_annotations.json")]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tbv_root": str(tbv_root), "argotweak_root": str(at_root),
        "scope": "real_and_constructed" if args.scan_constructed else "real_only",
        "local_log_count": len(logs), "local_argotweak_file_count": len(at_paths),
        "buckets": buckets, "output_rows": {k: len(v) for k, v in files.items()},
        "missing_official_log_ids": missing_logs, "missing_argotweak_log_ids": missing_at,
        "unexpected_log_ids": sorted(set(logs) - set(tbv)),
        "tbv_event_types": dict(Counter(e["event"]["change_type"] for e in event_rows)),
        "metadata_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        "source_provenance": manifest,
        "limits": ["Scene selection only, not frame labels or visible-change verification.",
                   "Sensor file contents and completeness have not been validated.",
                   "Ground-truth labels are for curation/evaluation, not detector inputs.",
                   "Keep real test separate from constructed train/val; preserve raw codes and old/new fields."]
    }
    out.mkdir(parents=True)
    for name, content in files.items():
        write_jsonl(out / name, content)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "buckets": buckets,
                      "review_logs": len(files["review_logs.jsonl"]),
                      "missing_logs": len(missing_logs), "missing_annotations": len(missing_at)},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tbv-root", type=Path, default=Path("/Volumes/初棠/Argoverse 2"))
    parser.add_argument("--argotweak-root", type=Path, default=Path("/Volumes/初棠/ArgoTweak"))
    parser.add_argument("--metadata", type=Path, default=HERE / "metadata")
    parser.add_argument("--out", type=Path, default=HERE / "output_real_v1")
    parser.add_argument("--scan-constructed", action="store_true",
                        help="Also read the 799 constructed train+val annotations (slower; not needed for first run)")
    try:
        run(parser.parse_args())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

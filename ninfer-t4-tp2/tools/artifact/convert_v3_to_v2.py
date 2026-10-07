#!/usr/bin/env python3
"""In-place converter from ninfer-v3 container header to canonical ninfer-v2."""

import json
import os
import struct
import sys

RESOURCE_MAP = {
    "resource/text/tokenizer.json": "frontend/tokenizer.json",
    "resource/text/tokenizer_config.json": "frontend/tokenizer_config.json",
    "resource/text/chat_template.jinja": "frontend/chat_template.jinja",
    "resource/text/generation_config.json": "frontend/generation_config.json",
    "resource/vision/preprocessor_config.json": "frontend/preprocessor_config.json",
    "resource/vision/video_preprocessor_config.json": "frontend/video_preprocessor_config.json",
}

FORMAT_MAP = {
    "bf16": "BF16",
    "fp32": "FP32",
    "int32": "I32",
    "q4_g64_fp16": "Q4G64_F16S",
    "q5_g64_fp16": "Q5G64_F16S",
    "q6_g64_fp16": "Q6G64_F16S",
    "q8_g32_fp16": "W8G32_F16S",
    "w8_g32_fp16": "W8G32_F16S",
}

LAYOUT_MAP = {
    "contiguous_le_v1": "contiguous-le-v1",
    "row_split_k128_v1": "row-split-k128-v1",
}

ENCODING_MAP = {
    "raw_bytes_v1": "raw-bytes-v1",
}


def get_v2_name(oid, parts_list):
    sorted_parts = sorted(parts_list, key=lambda x: x[1][0] if x[1] else 0)
    names = [p[0] for p in sorted_parts]
    first = names[0]
    parent = first.rsplit("/", 1)[0]
    suffixes = [n.rsplit("/", 1)[1] for n in names]
    comb = "_".join(suffixes)
    return f"{parent}/{comb}"


def convert_v3_to_v2(model_path: str):
    if not os.path.exists(model_path):
        print(f"Error: file not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    with open(model_path, "rb") as f:
        prefix = f.read(16)
        if len(prefix) < 16:
            print("Error: file too small for container header", file=sys.stderr)
            sys.exit(1)
        magic, json_bytes = struct.unpack("<8sQ", prefix)

    if magic == b"NINFER\x00\x02":
        print(f"Artifact {model_path} is already in canonical v2 container format.")
        return

    if magic != b"NINFER\x00\x03":
        print(f"Artifact magic {magic!r} is not supported by this tool.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading v3 header ({json_bytes} json bytes) from {model_path}...")
    with open(model_path, "rb") as f:
        f.seek(32)
        raw_json = f.read(json_bytes)

    decoder = json.JSONDecoder()
    v3, _ = decoder.raw_decode(raw_json.decode("utf-8"))

    obj_names = {}
    for bname, binfo in v3.get("bindings", {}).items():
        if "object" in binfo:
            obj_names[binfo["object"]] = bname

    part_map = {}
    for bname, binfo in v3.get("bindings", {}).items():
        if "parts" in binfo:
            for p in binfo["parts"]:
                part_map.setdefault(p["object"], []).append((bname, p.get("range")))

    for oid, plist in part_map.items():
        obj_names[oid] = get_v2_name(oid, plist)

    for o in v3.get("objects", []):
        if o.get("kind") == "resource" and o["id"] in RESOURCE_MAP:
            obj_names[o["id"]] = RESOURCE_MAP[o["id"]]

    if "weight/001116" in obj_names:
        obj_names["weight/001116"] = "text/draft_head"
    if "weight/001117" in obj_names:
        obj_names["weight/001117"] = "text/draft_head_token_ids"

    for oid, name in list(obj_names.items()):
        if name.startswith("mtp/layers/0/"):
            obj_names[oid] = name.replace("mtp/layers/0/", "mtp/layer/")

    for oid, name in list(obj_names.items()):
        if "attention/query_key_value" in name:
            obj_names[oid] = name.replace("attention/query_key_value", "attention/qkv")
        elif "attention/query_bias_key_bias_value_bias" in name:
            obj_names[oid] = name.replace(
                "attention/query_bias_key_bias_value_bias", "attention/qkv_bias"
            )
        elif "norm1_bias" in name:
            obj_names[oid] = name.replace("norm1_bias", "norm1/bias")
        elif "norm1_weight" in name:
            obj_names[oid] = name.replace("norm1_weight", "norm1/weight")
        elif "norm2_bias" in name:
            obj_names[oid] = name.replace("norm2_bias", "norm2/bias")
        elif "norm2_weight" in name:
            obj_names[oid] = name.replace("norm2_weight", "norm2/weight")
        elif "merger/norm_weight" in name:
            obj_names[oid] = name.replace("merger/norm_weight", "merger/norm/weight")
        elif "merger/norm_bias" in name:
            obj_names[oid] = name.replace("merger/norm_bias", "merger/norm/bias")

    v2_objects = []
    for o in v3.get("objects", []):
        vo = dict(o)
        if o["id"] not in obj_names:
            print(f"Warning: object {o['id']} has no binding name, using id", file=sys.stderr)
            vo["name"] = o["id"]
        else:
            vo["name"] = obj_names[o["id"]]
        del vo["id"]
        if "format" in vo:
            vo["format"] = FORMAT_MAP.get(vo["format"], vo["format"])
        if "layout" in vo:
            vo["layout"] = LAYOUT_MAP.get(vo["layout"], vo["layout"])
        if "encoding" in vo:
            vo["encoding"] = ENCODING_MAP.get(vo["encoding"], vo["encoding"])
        v2_objects.append(vo)

    v2_objects.sort(key=lambda x: x["offset"])

    v2_root = {
        "identity": {
            "model_id": v3.get("identity", {}).get("model_id", "qwen3.6-27b"),
            "weights_id": v3.get("identity", {}).get("weights_id", "groupwise-int"),
        },
        "objects": v2_objects,
    }

    v2_json = json.dumps(v2_root, separators=(",", ":"))
    v2_json_bytes = v2_json.encode("utf-8")

    TOTAL_HEADER_SIZE = 348160
    PREFIX_SIZE = 16
    AVAILABLE_JSON_SIZE = TOTAL_HEADER_SIZE - PREFIX_SIZE

    padding_size = AVAILABLE_JSON_SIZE - len(v2_json_bytes)
    if padding_size < 0:
        print(f"Error: v2 JSON ({len(v2_json_bytes)} B) exceeds available header size ({AVAILABLE_JSON_SIZE} B)", file=sys.stderr)
        sys.exit(1)

    full_json_payload = v2_json_bytes + (b" " * padding_size)
    new_prefix = struct.pack("<8sQ", b"NINFER\x00\x02", AVAILABLE_JSON_SIZE)
    new_header = new_prefix + full_json_payload
    assert len(new_header) == TOTAL_HEADER_SIZE

    print(f"Writing updated v2 header in-place ({TOTAL_HEADER_SIZE} bytes)...")
    with open(model_path, "r+b") as f:
        f.seek(0)
        f.write(new_header)

    print(f"Successfully converted container header in-place to v2 format.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/models/qwen3_6_27b.ninfer"
    convert_v3_to_v2(target)

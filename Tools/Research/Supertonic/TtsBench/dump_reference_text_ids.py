"""Dump Supertonic text ids from the reference implementation, for cross-checking.

The C# front end (LiteRtLmSupertonicText) has to produce exactly these ids, or
the model receives different input than the Python bench that validated it. This
writes the ground truth; Test-SupertonicTextParity.ps1 compares against it.

Usage:
  python dump_reference_text_ids.py --assets-dir <onnx dir> --out ids.json \
      [--text "..." --lang ko]...
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supertonic_helper as helper


DEFAULT_CASES = [
    ("고도 백 미터로 상승합니다.", "ko"),
    ("배터리 잔량 칠십 퍼센트.", "ko"),
    ("경고. 강풍이 감지되었습니다. 고도를 낮춥니다.", "ko"),
    ("귀환을 시작합니다. 예상 소요 시간 삼 분", "ko"),          # no final period: exercises the auto-period rule
    ("Ascending to one hundred meters.", "en"),
    ("Battery at 70% — check e.g., the pack", "en"),            # em dash, expression replacement, auto-period
    ("여기 [중요] 지점 / 확인 요망", "ko"),                      # bracket and slash to space
    ("“인용” 그리고 ‘작은’ 따옴표", "ko"),                        # curly quotes
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--lang", action="append", default=[])
    args = parser.parse_args()

    if args.text:
        cases = list(zip(args.text, args.lang or ["ko"] * len(args.text)))
    else:
        cases = DEFAULT_CASES

    processor = helper.load_text_processor(args.assets_dir)
    records = []
    for text, lang in cases:
        ids, mask = processor([text], [lang])
        records.append({
            "text": text,
            "lang": lang,
            "prepared": processor._preprocess_text(text, lang),
            "ids": [int(v) for v in ids[0].tolist()],
            "mask_len": int(mask.shape[-1]),
        })

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=1)

    print(f"wrote {len(records)} cases to {args.out}")
    for record in records:
        print(f"  {record['lang']}  ids={len(record['ids'])}  {record['text'][:40]}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())

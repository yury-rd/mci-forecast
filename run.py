"""Run the synthetic example, summarize saved results, or fit local inputs."""

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare connectivity features for MCI forecasting.")
    parser.add_argument("--mode", choices=("demo", "summary"), default="demo",
                        help="fit the synthetic example (default), or read the saved study results")
    parser.add_argument("--input", type=Path, help="local NPZ arrays to fit in place of the synthetic example")
    parser.add_argument("--splits", type=Path, help="JSON split plan; required with --input")
    parser.add_argument("--output", type=Path, help="write JSON here; otherwise print it")
    args = parser.parse_args(argv)
    if args.mode == "summary" and (args.input or args.splits):
        parser.error("--mode summary cannot be combined with --input or --splits")
    if args.input and not args.splits:
        parser.error("--input requires --splits for the same participant row order")

    try:
        if args.mode == "summary":
            result = json.loads((DATA / "results.json").read_text())
        else:
            import numpy as np
            from threadpoolctl import threadpool_limits
            from forecast import evaluate, synthetic_data

            if args.input:
                with np.load(args.input, allow_pickle=False) as arrays:
                    data = {key: arrays[key] for key in arrays.files}
                scope = "local cohort"
            else:
                data, scope = synthetic_data(), "synthetic example"
            splits = json.loads((args.splits or DATA / "splits.json").read_text())
            with threadpool_limits(limits=1):
                result = {"scope": scope, **evaluate(data, splits)}
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
            print(f"Saved {args.output}")
        else:
            print(payload, end="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

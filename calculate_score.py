"""Compute CER/WER on the MultiClin benchmark across every script-evaluation mode.

For a given model's inference result CSV, this iterates over all
(medical x number x unit) combinations of {original, target, both}, dynamically
resolves the multiscript reference for each utterance, and reports the resulting
CER/WER. This reproduces the per-mode result tables in the paper.
"""

from utils import *


def parse_score_args():
    parser = argparse.ArgumentParser(
        description="Compute CER/WER on MultiClin across all script-evaluation modes"
    )
    parser.add_argument("--lang", type=str, default="ko", choices=['ko', 'ja', 'zh', 'ar'],
                        help="Target language (ko, ja, zh, ar)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name as it appears in the result CSV filename")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory containing the inference result CSVs")
    parser.add_argument("--data_root", type=str, default="",
                        help="Dataset directory (contains labels.csv). Defaults to "
                             "interspeech26_multiscript_dataset_<lang>; downloaded automatically if missing.")
    parser.add_argument("--result_csv", type=str, default=None,
                        help="Explicit result CSV path. Defaults to the "
                             "Med-both_Num-both_Unit-both file for --model.")
    return parser.parse_args()


def main():
    args = parse_score_args()

    # The hypotheses are mode-independent, so any result CSV for the model works;
    # default to the fully multiscript-aware one.
    if args.result_csv is not None:
        csv_path = args.result_csv
    else:
        csv_path = os.path.join(
            args.results_dir,
            f"result_{args.model}_{args.lang}_Med-both_Num-both_Unit-both.csv",
        )

    if not os.path.exists(csv_path):
        print(f"Error: result CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    print(f"Loaded {len(df)} hypotheses from {csv_path}")

    # Download the dataset if needed and load the reference labels.
    data_root = ensure_dataset(args.lang, args.data_root or None)
    label_path = os.path.join(data_root, "labels.csv")
    if not os.path.exists(label_path):
        print(f"Error: label CSV not found at {label_path}")
        return
    label = pd.read_csv(label_path)

    target_col = 'multiscript'

    for medical in ['original', 'target', 'both']:
        for number in ['original', 'target', 'both']:
            for unit in ['original', 'target', 'both']:

                all_refs = []
                all_hyps = []

                for _, row in label.iterrows():
                    src = row['src']
                    script_raw = row[target_col]

                    matches = df.loc[df['src'] == src, 'hypothesis'].values
                    if len(matches) == 0:
                        continue
                    hyp_norm = matches[0]

                    ref_raw_norm = normalize_text(script_raw)

                    # For Arabic, normalize the full reference/hypothesis before scoring.
                    if args.lang == 'ar':
                        hyp_norm = normalize_arabic(hyp_norm)
                        ref_raw_norm = normalize_arabic(ref_raw_norm)

                    final_reference = clean_tags_by_priority(
                        ref_raw_norm, hyp_norm, medical, number, unit, args.lang
                    )

                    all_refs.append(final_reference)
                    all_hyps.append(hyp_norm)

                total_wer = wer(all_refs, all_hyps)
                total_cer = cer(all_refs, all_hyps)

                print(f"medical: {medical:8}, number: {number:8}, unit: {unit:8}  "
                      f"CER: {total_cer:.4f}, WER: {total_wer:.4f}")
            print("-" * 32)


if __name__ == "__main__":
    main()

from utils import *

from qwen_asr import Qwen3ASRModel


def main():
    args = parse_args()

    # Fix seeds for reproducibility.
    random.seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = str(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(42)

    # Download the dataset for this language if it is not already present locally.
    data_root = ensure_dataset(args.lang, args.data_root or None)

    target_col = 'multiscript'
    audio_dir = os.path.join(data_root, 'audios')
    csv_path = os.path.join(data_root, 'labels.csv')

    print(f"=== Configuration: Language [{args.lang}] ===")
    print(f"Target Column: {target_col}")
    print(f"Audio Directory: {audio_dir}")
    print("-" * 40)

    os.makedirs(args.output_dir, exist_ok=True)

    if 'Qwen/' not in args.model:
        args.model = f'Qwen/{args.model}'

    print(f"Loading Qwen3 ASR model: {args.model}...")

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,  # Cap batch size to avoid OOM. -1 means unlimited.
        max_new_tokens=65536,         # Allow long-form audio output.
    )

    language = LANG_CONFIG[args.lang]['qwen_lang']

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        print(f"Error: Column '{target_col}' not found in CSV.")
        return

    print(f"Loaded {len(df)} samples.")

    option_suffix = f"_{args.lang}_Med-{args.medical}_Num-{args.number}_Unit-{args.unit}"
    safe_model_name = args.model.split("/")[-1].replace("/", "_")

    csv_filename = os.path.join(args.output_dir, f"result_{safe_model_name}{option_suffix}.csv")

    if os.path.exists(csv_filename):
        inference_df = pd.read_csv(csv_filename)
        inference_df.dropna(inplace=True)
        results_df = inference_df.to_dict(orient='list')
    else:
        results_df = {
            'src': [],
            'filename': [],
            'best_reference': [],
            'hypothesis': []
        }

    print("Starting evaluation...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_raw = row['src']

        if row['src'] not in results_df['src']:
            script_raw = row[target_col]

            if pd.isna(script_raw):
                continue

            real_file_name = src_raw.replace('/', '_')
            filepath = os.path.join(audio_dir, f"{real_file_name}.wav")

            if not os.path.exists(filepath):
                if _ < 5:
                    print(f"[Warning] File not found: {filepath}")
                continue

            try:
                results = model.transcribe(
                    audio=filepath,
                    language=language,
                )
                hypothesis = results[0].text

                print(f"File: {filepath}")
                print(f"Hypothesis: {hypothesis}")
                print("-" * 32)
            except Exception as e:
                print(f"Error processing {src_raw}: {e}")
                hypothesis = ""

            hyp_norm = normalize_text(hypothesis)
            ref_raw_norm = normalize_text(script_raw)

            # For Arabic, normalize the full reference/hypothesis before scoring.
            if args.lang == 'ar':
                hyp_norm = normalize_arabic(hyp_norm)
                ref_raw_norm = normalize_arabic(ref_raw_norm)

            final_reference = clean_tags_by_priority(ref_raw_norm, hyp_norm, args.medical, args.number, args.unit, args.lang)

            results_df['src'].append(src_raw)
            results_df['filename'].append(f"{real_file_name}.wav")
            results_df['best_reference'].append(final_reference)
            results_df['hypothesis'].append(hyp_norm)

    new_df = pd.DataFrame(results_df)
    new_df.dropna(inplace=True)
    new_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

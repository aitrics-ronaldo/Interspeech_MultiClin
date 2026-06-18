from utils import *

from pydub import AudioSegment
from google import genai
from google.genai import types


# Zero-shot transcription prompt per language.
GEMINI_PROMPTS = {
    'ko': """
                # Role
                You are a professional medical stenographer. Your goal is to transcribe medical audio into a clear list of sentences in Korean.

                # Context
                The audio is a medical consultation. It contains professional Korean medical terms and patient symptoms.

                # Task: Pure Transcription
                - Transcribe the audio into Korean **word-for-word**.
                - **Do NOT include speaker labels (e.g., "Staff", "Patient") or any prefixes.**
                - Do NOT summarize, omit, or paraphrase.
                - If a part is inaudible, use "[Inaudible]".
                - **Provide the result as a list of sentences.**

                # Output Format (JSON)
                {
                "transcript": [
                    "첫 번째 문장...",
                    "두 번째 문장...",
                    "세 번째 문장..."
                ]
                }
            """,
    'ja': """
                # Role
                You are a professional medical stenographer. Your goal is to transcribe medical audio into a clear list of sentences in Japanese.

                # Context
                The audio is a medical consultation in Japanese. It contains professional Japanese medical terms (Kanji), disease names, and patient symptoms.

                # Task: Pure Transcription
                - Transcribe the audio into Japanese **word-for-word**.
                - Ensure correct use of Kanji, Hiragana, and Katakana based on the context.
                - **Do NOT include speaker labels (e.g., "Staff", "Patient") or any prefixes.**
                - Do NOT summarize, omit, or paraphrase.
                - If a part is inaudible, use "[Inaudible]".
                - **Provide the result as a list of sentences.**

                # Output Format (JSON)
                {
                "transcript": [
                    "最初の文章が入ります。",
                    "二番目の文章が入ります。",
                    "三番目の文章が入ります。"
                ]
                }
            """,
    'zh': """
                # Role
                You are a professional medical stenographer. Your goal is to transcribe medical audio into a clear list of sentences in Chinese (Simplified).

                # Context
                The audio is a medical consultation in Chinese. It contains professional Chinese medical terms, diagnoses, and patient symptoms.

                # Task: Pure Transcription
                - Transcribe the audio into Chinese (Simplified) **word-for-word**.
                - Ensure correct use of Chinese characters (Hanzi) based on the medical context.
                - **Do NOT include speaker labels (e.g., "Staff", "Patient") or any prefixes.**
                - Do NOT summarize, omit, or paraphrase.
                - If a part is inaudible, use "[Inaudible]".
                - **Provide the result as a list of sentences.**

                # Output Format (JSON)
                {
                "transcript": [
                    "这是第一句话。",
                    "这是第二句话。",
                    "这是第三句话。"
                ]
                }
            """,
    'ar': """
                # Role
                You are a professional medical stenographer. Your goal is to transcribe medical audio into a clear list of sentences in Arabic.

                # Context
                The audio is a medical consultation in Arabic. It contains specialized medical terminology and patient symptoms.
                The speakers may use Modern Standard Arabic (MSA) or regional dialects.

                # Task: Pure Transcription
                - Transcribe the audio into Arabic **word-for-word**.
                - Keep the transcription in the original dialect or style as spoken (Modern Standard Arabic or Ammiya).
                - **Do NOT include speaker labels (e.g., "Staff", "Patient") or any prefixes.**
                - Do NOT summarize, omit, or paraphrase.
                - If a part is inaudible, use "[Inaudible]".
                - **Provide the result as a list of sentences.**

                # Output Format (JSON)
                {
                "transcript": [
                    "الجملة الأولى هنا...",
                    "الجملة الثانية هنا...",
                    "الجملة الثالثة هنا..."
                ]
                }
            """,
}


def main():
    args = parse_args()

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

    print(f"Loading Gemini API: {args.model}...")
    try:
        client = genai.Client(api_key=args.api_key)

        # Clear any previously uploaded files.
        all_files = list(client.files.list())
        for f in tqdm(all_files, total=len(all_files)):
            client.files.delete(name=f.name)

        prompt_text = GEMINI_PROMPTS[args.lang]

    except Exception as e:
        print(f"Error loading Gemini API: {e}")
        return

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
        print(f"Found existing results: {len(inference_df)} rows")
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

            filepath = os.path.join(audio_dir, f"{src_raw.replace('/', '_')}.wav")

            if not os.path.exists(filepath):
                if _ < 5:
                    print(f"[Warning] File not found: {filepath}")
                continue

            # Re-encode to 16kHz mono 16-bit WAV before upload.
            output_file = './temp_audio.wav'
            audio = AudioSegment.from_file(filepath)
            optimized_audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            optimized_audio.export(output_file, format="wav")

            try:
                audio_file = client.files.upload(file=output_file)
                response = client.models.generate_content(
                    model=args.model,
                    contents=[
                        audio_file,
                        prompt_text
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0  # Deterministic transcription.
                    )
                )
                data = json.loads(response.text)
                if type(data) == dict:
                    try:
                        sentence_list = data.get("transcript", [])
                    except:
                        sentence_list = []
                        for ele in data:
                            sentence_list.append(ele.get("text", ""))
                elif type(data) == list:
                    sentence_list = data
                hypothesis = " ".join(sentence_list)

                time.sleep(2)
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
            results_df['filename'].append(f"{src_raw.replace('/', '_')}.wav")
            results_df['best_reference'].append(final_reference)
            results_df['hypothesis'].append(hyp_norm)

    new_df = pd.DataFrame(results_df)
    new_df.dropna(inplace=True)
    new_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

# # # # """
# # # # Gradio app to collect voice + transcript and upload to a Hugging Face dataset repo.

# # # # How it works (high level):
# # # #  - User records audio in the browser (Gradio Audio component).
# # # #  - User types the transcript.
# # # #  - On submit, the audio is saved as WAV, a JSONL manifest entry is created/appended,
# # # #    and both audio + updated manifest are uploaded to the Hugging Face repo using huggingface_hub.

# # # # Before running:
# # # #  - pip install -r requirements.txt
# # # #  - export HF_TOKEN="your_hf_token"  (or put in a .env file)
# # # #  - Set REPO_ID to "<username>/<repo-name>" (create this repo on HF Hub, repo type: dataset or dataset+space)




# import os
# import uuid
# import json
# import io
# from datetime import datetime
# from dotenv import load_dotenv
# import gradio as gr
# import soundfile as sf
# from huggingface_hub import upload_file

# # Load environment variables
# load_dotenv()
# HF_TOKEN = os.environ.get("HF_TOKEN")
# REPO_ID = os.environ.get("HF_REPO_ID")
# HF_HUB_HTTP_TIMEOUT = os.environ.get("HF_HUB_HTTP_TIMEOUT")
# REPO_SUBFOLDER = os.environ.get("HF_REPO_SUBFOLDER", "data")
# MANIFEST_FILENAME = "dataset.jsonl"

# if not HF_TOKEN or not REPO_ID:
#     raise EnvironmentError("HF_TOKEN and HF_REPO_ID must be set in .env")


# def process_and_upload(audio_array, sample_rate, transcript, speaker_id=None, language=None):
#     entry_id = str(uuid.uuid4())
#     timestamp = datetime.utcnow().isoformat() + "Z"
#     audio_filename = f"{REPO_SUBFOLDER}/{entry_id}.wav"
#     json_filename = f"{REPO_SUBFOLDER}/{entry_id}.json"

#     # --- Audio to BytesIO (no local file) ---
#     buf = io.BytesIO()
#     sf.write(buf, audio_array, samplerate=sample_rate, subtype='PCM_16', format="WAV")
#     buf.seek(0)

#     # Upload audio directly from memory
#     upload_file(
#         path_or_fileobj=buf,
#         path_in_repo=audio_filename,
#         repo_id=REPO_ID,
#         token=HF_TOKEN,
#         repo_type="dataset"
#     )

#     # --- Metadata per-record JSON in memory ---
#     meta = {
#         "id": entry_id,
#         "audio": audio_filename,
#         "transcript": transcript,
#         "speaker_id": speaker_id,
#         "language": language,
#         "timestamp": timestamp
#     }
#     meta_bytes = io.BytesIO(json.dumps(meta, ensure_ascii=False).encode("utf-8"))

#     # Upload per-record JSON directly
#     upload_file(
#         path_or_fileobj=meta_bytes,
#         path_in_repo=json_filename,
#         repo_id=REPO_ID,
#         token=HF_TOKEN,
#         repo_type="dataset"
#     )

#     # --- Aggregate manifest from HF (optional) ---
#     # For large datasets, you can build manifest separately using the HF API
#     # or run a batch script later for casting and manifest aggregation.

#     return True, f"Uploaded: {entry_id}"


# def on_submit(audio_path, transcript, speaker_id="", language=""):
#     if audio_path is None or not transcript.strip():
#         return "Please record audio and type transcript."
#     arr, sr = sf.read(audio_path, dtype="float32")
#     success, msg = process_and_upload(arr, sr, transcript, speaker_id, language)
#     return msg


# # --- Gradio UI ---
# with gr.Blocks() as demo:
#     gr.Markdown("## Voice + Transcript Collector\nRecord your voice, type transcript, submit to Hugging Face dataset.")
    
#     audio_input = gr.Audio(label="Record audio", type="filepath")
#     transcript_input = gr.Textbox(label="Transcript")
#     speaker_input = gr.Textbox(label="Speaker ID (optional)")
#     language_input = gr.Textbox(label="Language (optional)")
#     status_output = gr.Textbox(label="Status", interactive=False)
#     submit_btn = gr.Button("Submit")

#     submit_btn.click(
#         fn=on_submit,
#         inputs=[audio_input, transcript_input, speaker_input, language_input],
#         outputs=[status_output]
#     )

# demo.launch(share=True)

















import os
import uuid
import json
import io
import time
from datetime import datetime
from dotenv import load_dotenv
import gradio as gr
import soundfile as sf
from huggingface_hub import HfApi

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("HF_REPO_ID")
REPO_SUBFOLDER = os.environ.get("HF_REPO_SUBFOLDER", "data")
MANIFEST_FILENAME = "dataset.jsonl"

if not HF_TOKEN or not REPO_ID:
    raise EnvironmentError("HF_TOKEN and HF_REPO_ID must be set in .env")

# -------------------------
# Hugging Face API client
# -------------------------
# HF_HUB_HTTP_TIMEOUT from .env will be automatically used
hf = HfApi()

# -------------------------
# Safe upload function
# -------------------------
def safe_upload(fileobj, path_in_repo):
    """Retry HuggingFace uploads to avoid timeout crashes."""
    for attempt in range(3):
        try:
            return hf.upload_file(
                path_or_fileobj=fileobj,
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                token=HF_TOKEN,
                repo_type="dataset"
            )
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Upload failed, retrying... ({attempt+1}/3)")
            time.sleep(2)  # short wait before retry

# -------------------------
# Main process and upload
# -------------------------
def process_and_upload(audio_array, sample_rate, transcript, speaker_id=None, language=None):
    entry_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"
    audio_filename = f"{REPO_SUBFOLDER}/{entry_id}.flac"
    json_filename = f"{REPO_SUBFOLDER}/{entry_id}.json"

    # --- Convert audio to FLAC to reduce size and prevent timeout ---
    buf = io.BytesIO()
    sf.write(buf, audio_array, samplerate=sample_rate, format="FLAC")
    buf.seek(0)

    # Upload audio safely
    safe_upload(buf, audio_filename)

    # --- Prepare metadata JSON ---
    meta = {
        "id": entry_id,
        "audio": audio_filename,
        "transcript": transcript,
        "speaker_id": speaker_id,
        "language": language,
        "timestamp": timestamp
    }
    meta_bytes = io.BytesIO(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    meta_bytes.seek(0)

    # Upload metadata safely
    safe_upload(meta_bytes, json_filename)

    return True, f"Uploaded: {entry_id}"

# -------------------------
# Gradio submit function
# -------------------------
def on_submit(audio_path, transcript, speaker_id="", language=""):
    if audio_path is None or not transcript.strip():
        return "Please record audio and type transcript."
    arr, sr = sf.read(audio_path, dtype="float32")
    success, msg = process_and_upload(arr, sr, transcript, speaker_id, language)
    return msg

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Voice + Transcript Collector\nRecord your voice, type transcript, submit to Hugging Face dataset.")

    audio_input = gr.Audio(label="Record audio", type="filepath")
    transcript_input = gr.Textbox(label="Transcript")
    speaker_input = gr.Textbox(label="Speaker ID (optional)")
    language_input = gr.Textbox(label="Language (optional)")
    status_output = gr.Textbox(label="Status", interactive=False)
    submit_btn = gr.Button("Submit")

    submit_btn.click(
        fn=on_submit,
        inputs=[audio_input, transcript_input, speaker_input, language_input],
        outputs=[status_output]
    )

demo.launch(share=True)

# casting.py
# Script to cast the "audio" column of your Hugging Face dataset
# for proper playable audio in the data card.

import os
import shutil
import logging
from datasets import load_dataset, Audio 
from datasets import unresolve_features # FIX: Import context manager from top-level datasets module
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError 

# --- STANDARD PYTHON LOGGING SETUP ---
# This will output detailed debug information to the console to help diagnose download issues.
logging.basicConfig(level=logging.INFO)
huggingface_logger = logging.getLogger("huggingface_hub")
huggingface_logger.setLevel(logging.DEBUG)


# --- IMPORTANT: CONFIGURATION ---
REPO_ID = "IbraahimLab/Voice-so-data" 
# FIX: Removed invisible character U+00A0
REPO_SUBFOLDER = "tmp_audio_root" # Temporary folder for Hub download cache
REPO_LOCAL_STORAGE = os.path.join(REPO_SUBFOLDER, REPO_ID.split('/')[-1]) 
LOCAL_CWD_DATA_DIR = "data" # The folder where the push function expects to find the audio files

# --- DATASET PROCESSING ---
def process_dataset():
    # Make sure temporary folder exists and the local data dir exists
    os.makedirs(REPO_LOCAL_STORAGE, exist_ok=True)
    os.makedirs(LOCAL_CWD_DATA_DIR, exist_ok=True)
    
    try:
        print(f"Attempting to load dataset: {REPO_ID}")
        # NOTE: Authentication token is implicitly used here
        dataset = load_dataset(REPO_ID)
    except RepositoryNotFoundError as e:
        print("FATAL ERROR: Could not load repository. Check spelling or permissions.")
        shutil.rmtree(REPO_SUBFOLDER, ignore_errors=True)
        return 

    # Download missing audio files from the Hub
    print("Downloading audio files temporarily for casting...")
    
    downloaded_paths = {}
    
    # Use unresolve_features to force iteration to return the raw string path,
    # preventing the Audio feature from attempting to decode the file (and failing the string methods).
    with unresolve_features(dataset.features):
        for split in dataset.keys():
            for row in dataset[split]:
                # This is now reliably the string path (e.g., 'data/file.wav')
                audio_filename_in_repo = row['audio']
                
                # 1. Define the expected absolute path of the downloaded file (where it will be placed by hf_hub_download)
                local_cache_path = os.path.join(REPO_LOCAL_STORAGE, audio_filename_in_repo.replace('/', os.sep))
                
                # Ensure the final target directory exists
                os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)

                # Use a combined key for the downloaded_paths dict to ensure uniqueness across splits
                key = f"{split}/{audio_filename_in_repo}"

                if not os.path.exists(local_cache_path):
                    try:
                        # Explicitly set repo_type="dataset" to resolve the 404 error
                        # hf_hub_download returns the absolute path to the downloaded file
                        downloaded_file_path = hf_hub_download(
                            repo_id=REPO_ID, 
                            filename=audio_filename_in_repo, 
                            local_dir=REPO_LOCAL_STORAGE,
                            repo_type="dataset" 
                        )
                        downloaded_paths[key] = downloaded_file_path
                    
                    except Exception as download_e:
                        print("\n" + "*"*50)
                        print(f"CRITICAL DOWNLOAD FAILED: {audio_filename_in_repo}")
                        print(f"ACTUAL EXCEPTION DETAIL: {download_e}")
                        print("*"*50 + "\n")
                        continue
                else:
                    downloaded_paths[key] = local_cache_path

    print("Download completed.")

    # Update dataset paths to point to local files AND copy files to CWD expected location
    print("Preparing local files and updating dataset paths...")
    updated_dataset = dataset.copy() 

    # Use unresolve_features again for the second iteration loop
    with unresolve_features(updated_dataset.features):
        for split in updated_dataset.keys():
            for i, row in enumerate(updated_dataset[split]):
                # This is now reliably the string path (e.g., 'data/file.wav')
                audio_filename_in_repo = row['audio']
                
                key = f"{split}/{audio_filename_in_repo}"
                
                if key in downloaded_paths:
                    source_path = downloaded_paths[key]
                    
                    # The absolute path the 'push_to_hub' function expects to find the local file at (CWD/data/file.wav)
                    expected_cwd_path = os.path.join(os.getcwd(), audio_filename_in_repo.replace('/', os.sep))

                    # Copy the file from the temporary cache to the expected CWD path (FIX for WinError 3)
                    try:
                        # Ensure the expected CWD path directory exists
                        os.makedirs(os.path.dirname(expected_cwd_path), exist_ok=True)
                        # Use copy2 to preserve metadata (useful for mtime)
                        shutil.copy2(source_path, expected_cwd_path) 
                    except Exception as copy_e:
                        print(f"WARNING: Failed to copy file to CWD: {expected_cwd_path}. Push may fail. Details: {copy_e}")
                        
                    # Update the dataset column to use the absolute path of the file we just copied.
                    # This path is required for the local Audio() feature to load the file successfully.
                    updated_dataset[split][i]['audio'] = os.path.abspath(expected_cwd_path)
                else:
                    # If the download failed, we skip updating the path.
                    pass 

    # Cast the "audio" column to proper Audio type
    print("Casting 'audio' column...")
    # Iterate over all splits to apply cast_column
    for split in updated_dataset.keys():
        updated_dataset[split] = updated_dataset[split].cast_column("audio", Audio())

    # Push the cast dataset back to the Hub
    print(f"Pushing dataset with cast audio back to Hugging Face repository: {REPO_ID}...")
    try:
        # Iterate over splits and push them individually (FIX for 'dict' object error in old versions)
        for split_name, dataset_split in updated_dataset.items():
            print(f"Pushing split '{split_name}'...")
            # Use split=split_name to correctly update the remote split
            dataset_split.push_to_hub(REPO_ID, split=split_name)
        
        print("Dataset successfully updated on Hugging Face!")
    except Exception as push_e:
        print("\n" + "="*70)
        print("FATAL PUSH ERROR: Failed to push to Hugging Face Hub.")
        print("This usually means your token lacks WRITE permission or there is a final path mismatch.")
        print(f"Details: {push_e}")
        print("="*70 + "\n")


    # Clean up temporary audio files
    shutil.rmtree(REPO_SUBFOLDER, ignore_errors=True)
    # Also clean up the files copied to the CWD data folder
    shutil.rmtree(LOCAL_CWD_DATA_DIR, ignore_errors=True) 
    print("Temporary files cleaned up.")

# Run the process
if __name__ == '__main__':
    process_dataset()
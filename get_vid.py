import subprocess
import os
import sys

def download_from_modal_volume(volume_name, remote_path, local_path, force=True):

    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

    command = f"modal volume get {volume_name} {remote_path} {local_path}"
    if force:
        command += " --force"
    
    try:
        subprocess.run(
            command, 
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"SUCCESS: Downloaded {remote_path} to {local_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to download file.")
        print(f"Command used: {command}")
        return False

def download_with_retry(volume_name, remote_path, local_path, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        print(f"Download attempt {attempt}/{max_attempts}...")
        success = download_from_modal_volume(volume_name, remote_path, local_path)
        if success or os.path.exists(local_path):
            return True
        print(f"Retrying...")
    return False

download_path = "./videos/output.mp4" 
success = download_with_retry("genai-results", "output.mp4", download_path)

if os.path.exists(download_path):
    file_size_mb = os.path.getsize(download_path) / (1024 * 1024)
    print(f"Video downloaded successfully at: {download_path}")
    print(f"File size: {file_size_mb:.2f} MB")
else:
    print(f"WARNING: Could not verify download. Check if file exists at {download_path}")
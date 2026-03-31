import requests
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8080"
ASR_ENDPOINT = f"{API_BASE_URL}/api/asr"


def send_audio_file(file_path: str):
    """Send audio file to ASR API."""

    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found - {file_path}")
        return None

    # Open file in binary mode
    with open(file_path, 'rb') as audio_file:
        # Prepare the file for upload
        files = {
            'file': (Path(file_path).name, audio_file, 'audio/wav')
        }

        # Send POST request
        print(f"Sending file: {file_path}")
        response = requests.post(ASR_ENDPOINT, files=files)

    # Check response
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Status: {response.status_code}")
        return result
    else:
        print(f"Error! Status: {response.status_code}")
        print(f"Response: {response.text}")
        return None


# Example usage
if __name__ == "__main__":
    # Send a single file
    # result = send_audio_file("tone/demo/audio_examples/audio_long.flac")
    result = send_audio_file("tone/demo/audio_examples/audio_short.flac")

    if result:
        print(f"Количество фраз: {result['num_phrases']}")
        print("\nРасшифровка:")
        for phrase in result['transcript']:
            print(f"  [{phrase['start_time']:.2f}s - {phrase['end_time']:.2f}s] {phrase['text']}")
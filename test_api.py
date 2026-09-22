import requests
import os

# Ensure the server is running on localhost:8000
url = "http://127.0.0.1:8000/notes/upload/"

headers = {
    "X-Device-ID": "test_device_123"
}

# Create a dummy image for testing
with open("test_image.jpg", "wb") as f:
    f.write(b"dummy image data")

files = {
    "file": ("test_image.jpg", open("test_image.jpg", "rb"), "image/jpeg")
}

try:
    print("Testing upload API...")
    response = requests.post(url, headers=headers, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error connecting to server: {e}")

finally:
    # Cleanup
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")

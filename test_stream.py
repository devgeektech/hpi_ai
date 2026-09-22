import requests
import json
import sys
import sys

# Change this path to the image you want to test!
IMAGE_PATH = "sample.jpeg"  

url = "http://127.0.0.1:8000/notes/upload_stream/"
headers = {
    "x-device-id": "test-device-123"
}

try:
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": f}
        print("Starting stream connection...")
        print("-" * 50)
        
        # stream=True keeps the connection open
        with requests.post(url, headers=headers, files=files, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        # Parse the JSON from the SSE stream
                        data_str = decoded_line[6:]
                        data = json.loads(data_str)
                        
                        if data.get("status") == "Complete":
                            print("\n✅ Final Result Received:")
                            print(json.dumps(data["result"], indent=2))
                        else:
                            # Print the progress updates as they arrive
                            print(f"⏳ {data.get('status')}")
                            sys.stdout.flush()
                            
except FileNotFoundError:
    print(f"❌ Error: Could not find image at '{IMAGE_PATH}'.")
    print("Please change the IMAGE_PATH variable inside this script to point to a real image on your computer.")

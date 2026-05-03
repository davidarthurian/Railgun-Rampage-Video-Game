import asyncio
import cv2 as cv
import numpy as np
import websockets
import json

async def video_client():
    uri = "ws://192.168.1.102:8765"  # Replace with your server URI
    camera_index = 0  # Ensure to use index 0 since it works
    cap = cv.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open video device with index {camera_index}")
        return

    try:
        async with websockets.connect(uri) as websocket:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame from camera")
                    break

                # Flip the frame horizontally
                frame = cv.flip(frame, 1)

                # Find blue objects and their centroids
                blue_centroid, frame_with_boxes = find_objects(frame)

                # Display the video feed
                cv.imshow('Video Feed', frame_with_boxes)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                if blue_centroid:
                    json_data = json.dumps({"centroid": blue_centroid})
                    await websocket.send(json_data)
                    await asyncio.sleep(0.05)  # Prevents spamming the server too quickly

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()

def find_objects(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([150, 255, 255])
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

    return process_color(frame, blue_mask, (255, 0, 0))

def process_color(frame, mask, color):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    centroid = None
    if contours:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        centroid = [x + w // 2, y + h // 2]
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return centroid, frame

asyncio.run(video_client())
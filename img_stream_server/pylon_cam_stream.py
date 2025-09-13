from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
import io
import cv2
import os

# Try to import pypylon for Basler camera support
try:
    from pypylon import pylon
    BASLER_AVAILABLE = True
except ImportError:
    BASLER_AVAILABLE = False
    print("pypylon not available. Only webcam support enabled.")

app = FastAPI()

# ---- Camera initialization ----
camera = None
use_basler = False

def initialize_camera():
    global camera, use_basler

    # Try to initialize Basler camera first if available
    if BASLER_AVAILABLE:
        try:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # Setup converter for Basler camera
            global converter
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            use_basler = True
            print("Basler camera initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize Basler camera: {e}")

    # Fall back to webcam
    try:
        camera = cv2.VideoCapture(0)  # Use default webcam
        if not camera.isOpened():
            raise Exception("Could not open webcam")

        # Set some basic properties for better quality
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 30)

        use_basler = False
        print("Webcam initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize webcam: {e}")
        return False

def get_frame_basler():
    """Get frame from Basler camera"""
    grab = camera.RetrieveResult(5000)
    if not grab.GrabSucceeded():
        grab.Release()
        return None

    img = converter.Convert(grab).GetArray()
    grab.Release()
    return img

def get_frame_webcam():
    """Get frame from webcam"""
    ret, frame = camera.read()
    if not ret:
        return None
    return frame

@app.get("/frame")
def get_frame():
    if camera is None:
        return Response(content="Camera not initialized", status_code=500)

    # Get frame based on camera type
    if use_basler:
        img = get_frame_basler()
    else:
        img = get_frame_webcam()

    if img is None:
        return Response(content="No frame available", status_code=500)

    # Encode as JPEG
    _, jpeg = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")

@app.get("/camera-info")
def get_camera_info():
    """Get information about the current camera"""
    if camera is None:
        return {"status": "No camera initialized"}

    return {
        "camera_type": "Basler" if use_basler else "Webcam",
        "basler_available": BASLER_AVAILABLE,
        "status": "active"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize camera on startup"""
    if not initialize_camera():
        print("Warning: No camera could be initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up camera resources on shutdown"""
    global camera
    if camera is not None:
        if use_basler:
            camera.StopGrabbing()
            camera.Close()
        else:
            camera.release()
        print("Camera resources cleaned up")

if __name__ == "__main__":
    # Run FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8554)

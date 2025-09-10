from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
import io
import cv2
from pypylon import pylon

app = FastAPI()

# ---- Initialize Basler camera ----
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

@app.get("/frame")
def get_frame():
    # Grab one image
    grab = camera.RetrieveResult(5000)
    if not grab.GrabSucceeded():
        grab.Release()
        return Response(content="No frame", status_code=500)

    img = converter.Convert(grab).GetArray()
    grab.Release()

    # Encode as JPEG
    _, jpeg = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    # Run FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8554)
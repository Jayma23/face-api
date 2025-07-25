
---

### `app/main.py`

```python
from fastapi import FastAPI, UploadFile, File
from app.face_utils import analyze_face

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = analyze_face(image_bytes)
    return result


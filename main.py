from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes.annotate_route import router as annotate_router

app = FastAPI(
    title="Eye + Posture Annotator API",
    version="1.0",
    description="Upload video or use webcam → get annotated video + stats"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(annotate_router, prefix="/annotate", tags=["annotate"])

# Health Check
@app.get("/")
async def root():
    return {
        "message": "Eye + Posture Annotator API",
        "endpoints": {
            "POST /annotate": "Upload MP4/AVI or use webcam"
        }
    }
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
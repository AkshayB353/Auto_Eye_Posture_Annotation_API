import os
import uuid
import json
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import asyncio
from typing import Optional
import mediapipe as mp
from controller.process import process_video_source
from controller.evaluation import compute_f1_scores

router = APIRouter()

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

async def cleanup_files(file_paths: list[str], delay_sec: int = 5):
    await asyncio.sleep(delay_sec)
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"[CLEANUP] Removed: {path}")
            except Exception as e:
                print(f"[CLEANUP] Failed {path}: {e}")

@router.post("")
async def annotate_video(
    video_file: UploadFile = File(...),  
    grnd_truth: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None
):
    if not video_file or not video_file.filename:
        raise HTTPException(status_code=400, detail="video_file is required")

    filename = video_file.filename.lower()
    if not (filename.endswith(".mp4") or filename.endswith(".avi")):
        raise HTTPException(
            status_code=400,
            detail="Only .mp4 or .avi files are allowed"
        )

    if video_file.content_type not in ["video/mp4", "video/avi"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Only MP4 or AVI allowed"
        )
    
    if grnd_truth and grnd_truth.content_type != "application/json":
        raise HTTPException(status_code=400, detail="grnd_truth must be a JSON file")

    # Generate unique filenames
    uid = uuid.uuid4().hex[:8]
    base_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in video_file.filename.rsplit(".", 1)[0])
    ext = "mp4" if filename.endswith(".mp4") else "avi"
    input_name = f"{base_name}_{uid}"
    input_video_path = os.path.join(TEMP_DIR, f"{input_name}_input.{ext}")
    output_video_path = os.path.join(TEMP_DIR, f"{input_name}_output.mp4")

    try:
        content = await video_file.read()
        with open(input_video_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    ground_truth_data = None
    tmp_grnd = None
    files_to_cleanup = [input_video_path, output_video_path]

    if grnd_truth:
        try:
            grnd_content = await grnd_truth.read()
            ground_truth_data = json.loads(grnd_content)
            tmp_grnd = os.path.join(TEMP_DIR, f"{input_name}_grnd.json")
            with open(tmp_grnd, "wb") as f:
                f.write(grnd_content)
            files_to_cleanup.append(tmp_grnd)
        except json.JSONDecodeError:
            await cleanup_files(files_to_cleanup, delay_sec=0)
            raise HTTPException(status_code=400, detail="Invalid JSON in grnd_truth")
        except Exception as e:
            await cleanup_files(files_to_cleanup, delay_sec=0)
            raise HTTPException(status_code=500, detail=f"Failed to process grnd_truth: {str(e)}")

    try:
        result_json, video_path = await process_video_source(
            source=input_video_path,
            original_filename=video_file.filename,
            face_mesh=face_mesh,
            pose=pose,
            output_video_path=output_video_path,
        )

        generated_labels = result_json["labels_per_frame"]

        final_response = {
            "video_name": video_file.filename,
            "total_frames": result_json["total_frames"],
            "labels_per_frame": generated_labels
        }

        # Add F1 scores if ground truth provided 
        if ground_truth_data and "labels_per_frame" in ground_truth_data:
            try:
                f1_scores = compute_f1_scores(
                    ground_truth=ground_truth_data["labels_per_frame"],
                    generated=generated_labels
                )
                final_response.update({
                    "eye_f1": f1_scores["eye_f1"],
                    "posture_f1": f1_scores["posture_f1"]
                })
            except Exception as e:
                print(f"[F1 ERROR] {e}")

        return JSONResponse(
            content=jsonable_encoder(final_response),
            media_type="application/json",
            headers={"Content-Disposition": f"inline; filename=labels_{input_name}.json"}
        )

    except Exception as e:
        await cleanup_files(files_to_cleanup, delay_sec=0)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        background_tasks.add_task(cleanup_files, files_to_cleanup.copy(), delay_sec=5)
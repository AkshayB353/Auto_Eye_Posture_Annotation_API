from typing import Optional, Dict
import cv2
import numpy as np
from datetime import datetime
import asyncio
from controller.eye_state import init_eye_state, process_eye_frame
from controller.posture_state import (
    init_posture_state,
    calibrate_posture_frame,
    process_posture_frame,
)

# Backend-only – toggle for testing

SHOW_PROCESSING = False    # Set to False to disable cv2.imshow()
SAVE_OUTPUT_VIDEO = False    # Set to True to save annotated output video

async def process_video_source(
    source: str,
    original_filename: str,
    face_mesh,
    pose,
    output_video_path: Optional[str] = None,
) -> tuple[dict, str]:
    """
    Process uploaded video file:
    - Warm-up eye tracker
    - Calibrate posture
    - Annotate every frame
    - Return labels + optionally save annotated video
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception("Cannot open video source")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_writer = None
    if SAVE_OUTPUT_VIDEO and output_video_path:
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    eye_state = init_eye_state(fps=fps)
    posture_state = init_posture_state()

    stats = {
        "video_name": original_filename,
        "total_frames": 0,
        "blink_count": 0,
        "posture_scores": [],
        "calibrated": False,
        "warmed_up": False,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
    }

    labels_per_frame: Dict[int, Dict[str, str]] = {}

    # PHASE 1: Warm-up + Calibration 
    warmup_done = False
    calib_done = False

    while cap.isOpened() and not (warmup_done and calib_done):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = face_mesh.process(rgb)
        pose_res = pose.process(rgb)

        # eye (warm-up)
        eye_state, _ = process_eye_frame(frame, face_res, eye_state)
        warmup_done = eye_state["warmed_up"]

        # Calibrate posture
        posture_state, calib_done = calibrate_posture_frame(frame, face_res, pose_res, posture_state)

        stats["total_frames"] += 1

        if SHOW_PROCESSING:
            disp_frame = frame.copy()
            cv2.putText(disp_frame, "SETUP: Calibrating + Warming...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Backend Processing", disp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out_writer:
            out_writer.write(frame)

        await asyncio.sleep(0)

    stats["warmed_up"] = warmup_done
    stats["calibrated"] = calib_done

    # if calibration or warm-up failed 
    if not warmup_done:
        cap.release()
        if out_writer:
            out_writer.release()
        raise Exception("Eye warm-up failed, insufficient face data")

    if not calib_done:
        cap.release()
        if out_writer:
            out_writer.release()
        raise Exception("Posture calibration failed, could not detect reference pose")

    # from beginning for final pass ===
    cap.release()
    await asyncio.sleep(0.1)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # PHASE 2: Final Annotation Pass
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = face_mesh.process(rgb)
        pose_res = pose.process(rgb)

        eye_state, eye_info = process_eye_frame(frame, face_res, eye_state)
        posture_info = process_posture_frame(frame, face_res, pose_res, posture_state)

        # per-frame labels
        labels_per_frame[str(frame_no)] = {
            "eye_state": eye_info["status"],
            "posture": posture_info["status"]
        }

        # Update stats
        stats["blink_count"] = eye_info["blink_count"]
        stats["posture_scores"].append(posture_info["smooth_score"])

        # annotations on frame
        cv2.putText(frame, f"EAR: {eye_info['ear']:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 0), 2)
        eye_color = (0, 0, 255) if eye_info['eyes_closed'] else (0, 255, 0)
        cv2.putText(frame, f"Eye: {eye_info['status']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)
        cv2.putText(frame, f"Blinks: {eye_info['blink_count']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Posture: {posture_info['status']}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_info['color'], 2)
        cv2.putText(frame, f"Score: {posture_info['smooth_score']:.2f}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if SHOW_PROCESSING:
            cv2.imshow("Backend Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out_writer:
            out_writer.write(frame)

        frame_no += 1
        await asyncio.sleep(0)

    # Final stats
    stats["total_frames"] = frame_no
    stats["end_time"] = datetime.now().isoformat()

    if stats["posture_scores"]:
        scores = np.array(stats["posture_scores"])
        stats["avg_posture_score"] = round(float(scores.mean()), 3)
        stats["min_posture_score"] = round(float(scores.min()), 3)
        stats["max_posture_score"] = round(float(scores.max()), 3)

    result_json = {
        "video_name": original_filename,
        "total_frames": stats["total_frames"],
        "labels_per_frame": labels_per_frame
    }

    cap.release()
    if out_writer:
        out_writer.release()
    if SHOW_PROCESSING:
        cv2.destroyAllWindows()

    return result_json, output_video_path
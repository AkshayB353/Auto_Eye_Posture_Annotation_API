from typing import Dict, Any
from sklearn.metrics import f1_score
import numpy as np

def compute_f1_scores(
    ground_truth: Dict[str, Dict[str, str]],
    generated: Dict[str, Dict[str, str]]
) -> Dict[str, float]:
    """
    Compares ground truth and generated labels per frame.
    Handles string frame keys safely with numerical sorting.
    """
    if not ground_truth:
        return {"eye_f1": None, "posture_f1": None}

    # Convert string keys to int 
    gt_frames = sorted(ground_truth.keys(), key=int)
    gen_frames = sorted(generated.keys(), key=int)

    if len(gt_frames) != len(gen_frames):
        raise ValueError(
            f"Frame count mismatch: GT has {len(gt_frames)}, "
            f"Generated has {len(gen_frames)}"
        )

    # check all gt frames exist in generated
    missing = [f for f in gt_frames if f not in generated]
    if missing:
        raise ValueError(f"Missing generated labels for frames: {missing[:10]}")

    eye_gt = []
    eye_pred = []
    posture_gt = []
    posture_pred = []

    for frame in gt_frames:
        gt = ground_truth[frame]
        pred = generated[frame]
        eye_gt.append(gt.get("eye_state", ""))
        eye_pred.append(pred.get("eye_state", ""))
        posture_gt.append(gt.get("posture", ""))
        posture_pred.append(pred.get("posture", ""))

    # F1
    eye_f1 = f1_score(eye_gt, eye_pred, average='macro', zero_division=0)
    posture_f1 = f1_score(posture_gt, posture_pred, average='macro', zero_division=0)

    return {
        "eye_f1": round(float(eye_f1), 4),
        "posture_f1": round(float(posture_f1), 4)
    }
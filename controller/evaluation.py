from typing import Dict, Any
from sklearn.metrics import f1_score
import numpy as np

def compute_f1_scores(
    ground_truth: Dict[int, Dict[str, str]],
    generated: Dict[int, Dict[str, str]]
) -> Dict[str, float]:
    """
    Compares ground truth and generated labels per frame.
    Returns F1 score for eye_state and posture.
    """
    if not ground_truth:
        return {"eye_f1": None, "posture_f1": None}

    gt_frames = sorted(ground_truth.keys())
    gen_frames = sorted(generated.keys())

    # must have same frame count
    if len(gt_frames) != len(gen_frames):
        raise ValueError("Frame count mismatch between ground truth and generated")

    eye_gt = []
    eye_pred = []
    posture_gt = []
    posture_pred = []

    for frame in gt_frames:
        if frame not in generated:
            raise ValueError(f"Missing generated label for frame {frame}")

        gt = ground_truth[frame]
        pred = generated[frame]

        eye_gt.append(gt.get("eye_state", ""))
        eye_pred.append(pred.get("eye_state", ""))

        posture_gt.append(gt.get("posture", ""))
        posture_pred.append(pred.get("posture", ""))

    #F1 
    eye_f1 = f1_score(eye_gt, eye_pred, average='macro', zero_division=0)
    posture_f1 = f1_score(posture_gt, posture_pred, average='macro', zero_division=0)

    return {
        "eye_f1": round(float(eye_f1), 4),
        "posture_f1": round(float(posture_f1), 4)
    }
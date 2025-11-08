import numpy as np

# Config
EMA_ALPHA           = 0.3
CALIB_FRAMES_NEEDED = 60
MIN_FACE_PX         = 60
MAX_FACE_PX         = 800
MIN_SHOULDER_PX     = 40

GOOD_CENTER_REL     = -0.85
SENSITIVITY         = 5.0
SCALE_FACTOR        = 2.5
FACE_WEIGHT         = 0.7
SHOULDER_WEIGHT     = 0.3

DIST_MILD_MAX       = 50

FACE_WIDTH_AT_50CM_PX = 120.0
REAL_FACE_WIDTH_CM    = 14.0
FOCAL_LENGTH_PX       = (FACE_WIDTH_AT_50CM_PX * 50.0) / REAL_FACE_WIDTH_CM

# Landmark indices
CHIN, FOREHEAD, NOSE_TIP = 152, 10, 1
L_SH, R_SH = 11, 12
L_EYE_OUT, R_EYE_OUT = 33, 263

def init_posture_state():
    return {
        "face_ref": None,
        "shoulder_ref": None,
        "smooth_score": 1.25,
        "calibrated": False,
        "calib_face": [],
        "calib_shoulder": [],
    }

def _px(landmark, h, w):
    return int(landmark.x * w), int(landmark.y * h)

def _estimate_distance(flm, h, w):
    try:
        le = _px(flm[L_EYE_OUT], h, w)
        re = _px(flm[R_EYE_OUT], h, w)
        w_px = abs(le[0] - re[0])
        if w_px < 20: return None
        return (REAL_FACE_WIDTH_CM * FOCAL_LENGTH_PX) / w_px
    except:
        return None

def _status_from_distance(d):
    if d >= DIST_MILD_MAX:
        return "Straight", (0,255,0), 2.0
    else:
        return "Hunched", (0,0,255), 0.5
    

def calibrate_posture_frame(frame_bgr, face_results, pose_results, state):
    if not (face_results.multi_face_landmarks and pose_results.pose_landmarks):
        return state, False

    h, w = frame_bgr.shape[:2]
    flm = face_results.multi_face_landmarks[0].landmark
    plm = pose_results.pose_landmarks.landmark

    chin_y = _px(flm[CHIN], h, w)[1]
    fore_y = _px(flm[FOREHEAD], h, w)[1]
    face_h = abs(chin_y - fore_y)

    lsh = _px(plm[L_SH], h, w)
    rsh = _px(plm[R_SH], h, w)
    sh_w = abs(lsh[0] - rsh[0])

    if MIN_FACE_PX <= face_h <= MAX_FACE_PX and sh_w >= MIN_SHOULDER_PX:
        state["calib_face"].append(face_h)
        state["calib_shoulder"].append(sh_w)

    done = len(state["calib_face"]) >= CALIB_FRAMES_NEEDED
    if done and len(state["calib_face"]) >= 10:
        state["face_ref"] = np.median(state["calib_face"])
        state["shoulder_ref"] = np.median(state["calib_shoulder"])
        state["calibrated"] = True
        state["calib_face"] = state["calib_shoulder"] = []
    return state, done


def process_posture_frame(frame_bgr, face_results, pose_results, state):
    h, w = frame_bgr.shape[:2]
    info = {
        "status": "No face",
        "color": (200,200,200),
        "score": 0.0,
        "smooth_score": round(state["smooth_score"], 3),
        "calibrated": state["calibrated"],
    }

    if not face_results.multi_face_landmarks:
        return info

    flm = face_results.multi_face_landmarks[0].landmark
    nose_pt = _px(flm[NOSE_TIP], h, w)
    chin_y = _px(flm[CHIN], h, w)[1]
    fore_y = _px(flm[FOREHEAD], h, w)[1]
    face_h = abs(chin_y - fore_y)

    if (pose_results.pose_landmarks and MIN_FACE_PX <= face_h <= MAX_FACE_PX):
        plm = pose_results.pose_landmarks.landmark
        lsh = _px(plm[L_SH], h, w)
        rsh = _px(plm[R_SH], h, w)
        sh_w = abs(lsh[0] - rsh[0])
        mid_sh = ((lsh[0] + rsh[0]) // 2, (lsh[1] + rsh[1]) // 2)

        if sh_w >= MIN_SHOULDER_PX and state["calibrated"]:
            ref = FACE_WEIGHT * state["face_ref"] + SHOULDER_WEIGHT * state["shoulder_ref"]
            cur = FACE_WEIGHT * face_h + SHOULDER_WEIGHT * sh_w
            scale_ratio = ref / cur if cur > 0 else 1.0
            ratio = (nose_pt[1] - mid_sh[1]) / face_h * scale_ratio
            shifted = ratio - GOOD_CENTER_REL
            raw = SCALE_FACTOR * np.exp(-SENSITIVITY * shifted)
            raw = np.clip(raw, 0, SCALE_FACTOR)
            state["smooth_score"] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * state["smooth_score"]
            score = state["smooth_score"]

            if score >= 1.30:
                status, color = "Straight", (0,255,0)
            else:
                status, color = "Hunched", (0,0,255)

            info.update({
                "status": status, "color": color, "score": round(score, 3),
                "smooth_score": round(state["smooth_score"], 3),
                "nose_pt": nose_pt, "mid_shoulder": mid_sh,
            })
            return info
        else:
            d = _estimate_distance(flm, h, w)
            if d is not None:
                status, color, scr = _status_from_distance(d)
                state["smooth_score"] = EMA_ALPHA * scr + (1 - EMA_ALPHA) * state["smooth_score"]
                info.update({
                    "status": status, "color": color, "score": scr,
                    "smooth_score": round(state["smooth_score"], 3),
                    "distance_cm": round(d, 1)
                })
                return info

    if MIN_FACE_PX <= face_h <= MAX_FACE_PX:
        d = _estimate_distance(flm, h, w)
        if d is not None:
            status, color, scr = _status_from_distance(d)
            state["smooth_score"] = EMA_ALPHA * scr + (1 - EMA_ALPHA) * state["smooth_score"]
            info.update({
                "status": status, "color": color, "score": scr,
                "smooth_score": round(state["smooth_score"], 3),
                "distance_cm": round(d, 1)
            })
        else:
            info["status"], info["color"] = "Face too small", (200,200,200)
    else:
        info["status"], info["color"] = "Too close/far", (0,0,255)

    return info
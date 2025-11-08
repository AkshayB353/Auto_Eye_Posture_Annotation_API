import numpy as np
from collections import deque

# Config

MODE = "COMBINED" #3 Modes: HYSTERESIS , TIME, COMBINED

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

HISTORY_SEC        = 2.0
MIN_HISTORY_SEC    = 0.5
CLOSE_TIME_SEC     = 0.07
OPEN_TIME_SEC      = 0.10

THRESH_RATIO       = 0.70
CLOSE_THRESH_RATIO = 0.65
OPEN_THRESH_RATIO  = 0.75


def init_eye_state(fps: float = 30.0):
    fps = max(fps, 1.0)
    history_frames = max(1, int(fps * HISTORY_SEC))
    min_history = max(10, int(fps * MIN_HISTORY_SEC))
    return {
        "fps": fps,
        "history_frames": history_frames,
        "min_history": min_history,
        "consec_closed": max(1, int(fps * CLOSE_TIME_SEC)),
        "consec_open": max(1, int(fps * OPEN_TIME_SEC)),

        "ear_history": deque(maxlen=history_frames),
        "blink_count": 0,
        "eyes_closed": False,
        "closed_counter": 0,
        "open_counter": 0,
        "warmed_up": False,  # will set to True during calibration
    }

def _ear_from_landmarks(lm):
    def dist(a, b): return np.linalg.norm(lm[a] - lm[b])
    v1 = dist(LEFT_EYE[1], LEFT_EYE[5]) + dist(LEFT_EYE[2], LEFT_EYE[4])
    h1 = dist(LEFT_EYE[0], LEFT_EYE[3])
    v2 = dist(RIGHT_EYE[1], RIGHT_EYE[5]) + dist(RIGHT_EYE[2], RIGHT_EYE[4])
    h2 = dist(RIGHT_EYE[0], RIGHT_EYE[3])
    ear1 = v1 / (2 * h1) if h1 > 0 else 0
    ear2 = v2 / (2 * h2) if h2 > 0 else 0
    return (ear1 + ear2) / 2

def process_eye_frame(frame_bgr, face_results, state):
    """
    - Warm-up fills ear_history
    - Once warmed_up=True → full blink detection
    - Can re-enter warm-up if state["warmed_up"] = False
    """
    ear = 0.0
    info = {
        "ear": 0.0,
        "status": "No face",
        "eyes_closed": state["eyes_closed"],
        "blink_count": state["blink_count"],
        "blink_increment": 0,
        "mode": MODE,
        "warmed_up": state["warmed_up"],
        "history_len": len(state["ear_history"]),
    }

    if not face_results.multi_face_landmarks:
        return state, info

    lm_raw = face_results.multi_face_landmarks[0].landmark
    h, w = frame_bgr.shape[:2]
    lm = np.array([(p.x * w, p.y * h) for p in lm_raw])
    ear = _ear_from_landmarks(lm)
    info["ear"] = round(ear, 4)


    # WARM-UP PHASE
   
    if not state["warmed_up"]:
        state["ear_history"].append(ear)
        if len(state["ear_history"]) >= state["min_history"]:
            state["warmed_up"] = True
            info["status"] = "Warming Up... (Done)"
        else:
            info["status"] = f"Warming Up... ({len(state['ear_history'])}/{state['min_history']})"
        return state, info

    
    # BLINK DETECTION 
    
    open_median = np.median(state["ear_history"])

    if MODE == "TIME":
        thresh = open_median * THRESH_RATIO
        if ear < thresh:
            state["closed_counter"] += 1
            state["open_counter"] = 0
            if state["closed_counter"] >= state["consec_closed"]:
                state["eyes_closed"] = True
        else:
            state["open_counter"] += 1
            state["closed_counter"] = 0
            if state["open_counter"] >= state["consec_open"]:
                if state["eyes_closed"]:
                    state["blink_count"] += 1
                    info["blink_increment"] = 1
                state["eyes_closed"] = False
                state["ear_history"].append(ear)

    elif MODE == "HYSTERESIS":
        tc = open_median * CLOSE_THRESH_RATIO
        to = open_median * OPEN_THRESH_RATIO
        if ear < tc and not state["eyes_closed"]:
            state["eyes_closed"] = True
        elif ear > to and state["eyes_closed"]:
            state["eyes_closed"] = False
            state["blink_count"] += 1
            info["blink_increment"] = 1
            state["ear_history"].append(ear)
        elif not state["eyes_closed"] and ear > to:
            state["ear_history"].append(ear)

    elif MODE == "COMBINED":
        tc = open_median * CLOSE_THRESH_RATIO
        to = open_median * OPEN_THRESH_RATIO
        if not state["eyes_closed"]:
            if ear < tc:
                state["closed_counter"] += 1
                state["open_counter"] = 0
                if state["closed_counter"] >= state["consec_closed"]:
                    state["eyes_closed"] = True
            else:
                state["closed_counter"] = 0
                if ear > to:
                    state["ear_history"].append(ear)
        else:
            if ear > to:
                state["open_counter"] += 1
                state["closed_counter"] = 0
                if state["open_counter"] >= state["consec_open"]:
                    state["eyes_closed"] = False
                    state["blink_count"] += 1
                    info["blink_increment"] = 1
                    state["ear_history"].append(ear)
            else:
                state["open_counter"] = 0

    info["eyes_closed"] = state["eyes_closed"]
    info["blink_count"] = state["blink_count"]
    info["status"] = "Closed" if state["eyes_closed"] else "Open"

    return state, info
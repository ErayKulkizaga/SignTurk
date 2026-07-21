"""SignTurk FastAPI application.

The service combines authentication, persistence, live sign recognition,
sentence assembly, optional text-to-speech, and 3D avatar endpoints.
"""

import os, json, asyncio, time, traceback, uuid, logging, base64, shutil, tempfile, secrets
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── .env yükleme (HF_TOKEN vb. ortam değişkenleri) ────────
# Proje kökündeki .env dosyasını ortam değişkeni olarak okur; böylece
# her oturumda elle `export`/`set` yapmaya gerek kalmaz. python-dotenv
# kurulu değilse sessizce atlanır (yalnızca kolaylık katmanı, zorunlu değil).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from database import engine, get_db, Base, database_name
import models

# ── Landmark smoothing ───────────────────────────────────
try:
    from landmark_smoother import smooth_landmark_data
    SMOOTHER_AVAILABLE = True
except ImportError:
    SMOOTHER_AVAILABLE = False
    print("[UYARI] landmark_smoother modülü bulunamadı — /landmark endpoint ham veri döndürür")

# ── TensorFlow ────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
try:
    tf.get_logger().setLevel("ERROR")
except AttributeError:
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ═══════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("tsl")

# ═══════════════════════════════════════════════════════════
#  PASSWORD HASHING
# ═══════════════════════════════════════════════════════════
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def hash_pw(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_pw(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ═══════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR   = BASE_DIR / "static"
UPLOADS_DIR  = BASE_DIR / "uploads"

# Live recognition model
MODEL_DIR    = BASE_DIR / "demo_assets_179"

# Legacy animation/compatibility model
ASSETS_DIR   = BASE_DIR / "model_assets"

FRONTEND_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

UI_FILE = FRONTEND_DIR / "sign_turk_ui.html"

# ═══════════════════════════════════════════════════════════
#  GLOBALS — Live recognition model (179 classes)
# ═══════════════════════════════════════════════════════════
MODEL        = None
LABEL_MAP    = {}
NORM_MEAN    = None
NORM_STD     = None
DEMO_CONFIG  = {}
MP_HANDS     = None

SEQ_LEN      = 16
FEAT_DIM     = 156
NUM_CLASSES  = 179
CONFIDENCE_THRESHOLD = 0.40
MODEL_LOAD_TIME  = 0.0
AVG_INFERENCE_MS = 0.0
INFERENCE_COUNT  = 0

MIN_HAND_NORM = 1e-6
HISTORY_SAVE_COOLDOWN_SEC = 4.0
MEDIAPIPE_DETECTION_CONFIDENCE = 0.15
MEDIAPIPE_TRACKING_CONFIDENCE = 0.15
MEDIAPIPE_STATIC_IMAGE_MODE = False
MEDIAPIPE_MODEL_COMPLEXITY = 1
MAX_MISSED_HAND_FRAMES = 6
LIVE_BUFFER_STRIDE = 0

# ═══════════════════════════════════════════════════════════
#  GLOBALS — Legacy animation model (184 classes)
# ═══════════════════════════════════════════════════════════
ANIM_MODEL     = None
ANIM_LABEL_MAP = {}
ANIM_NORM_MEAN = None
ANIM_NORM_STD  = None
ANIM_SEQ_LEN   = 30
ANIM_SINGLE_DIM = 126
ANIM_CONF_THRESH = 0.5
ANIM_TOP_K       = 3

# Landmark dizini (3D animasyon için)
_candidate_landmark_dirs = [
    BASE_DIR / "dataset" / "landmarks",
    BASE_DIR.parent / "dataset" / "landmarks",
    BASE_DIR / "landmarks",
]
LANDMARKS_DIR  = None
LANDMARK_INDEX = {}

# ═══════════════════════════════════════════════════════════
#  MEDIAPIPE INIT
# ═══════════════════════════════════════════════════════════
def prepare_mediapipe_resource_path(mp_module):
    try:
        package_root = Path(mp_module.__file__).resolve().parent
        str(package_root).encode("ascii")
        return
    except UnicodeEncodeError:
        pass
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Resource path check failed: {e}")
        return

    try:
        from mediapipe.python import solution_base

        ascii_root = Path(tempfile.gettempdir()) / "tsl_mediapipe_site"
        modules_src = package_root / "modules"
        modules_dst = ascii_root / "mediapipe" / "modules"
        hand_graph = modules_dst / "hand_landmark" / "hand_landmark_tracking_cpu.binarypb"

        if not hand_graph.exists():
            modules_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(modules_src, modules_dst, dirs_exist_ok=True)

        fake_solution = ascii_root / "mediapipe" / "python" / "solution_base.py"
        fake_solution.parent.mkdir(parents=True, exist_ok=True)
        fake_solution.touch(exist_ok=True)
        solution_base.__file__ = str(fake_solution)
        logger.info(f"[MEDIAPIPE] Using ASCII resource root: {ascii_root}")
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] ASCII resource workaround failed: {e}")


def init_mediapipe():
    try:
        import mediapipe as mp
        prepare_mediapipe_resource_path(mp)
        hands = mp.solutions.hands.Hands(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_hands=2,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        hands.process(np.zeros((100, 100, 3), dtype=np.uint8))
        logger.info("[MEDIAPIPE] Strategy 1 OK")
        return hands
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Strategy 1 failed: {e}")

    try:
        from mediapipe.python.solutions import hands as mp_hands
        hands = mp_hands.Hands(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_hands=2,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        hands.process(np.zeros((100, 100, 3), dtype=np.uint8))
        logger.info("[MEDIAPIPE] Strategy 2 OK")
        return hands
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Strategy 2 failed: {e}")

    try:
        import importlib
        import mediapipe
        hands_module = getattr(mediapipe, "solutions", None)
        if hands_module is None:
            hands_module = importlib.import_module("mediapipe.solutions")
        hands = hands_module.hands.Hands(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_hands=2,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        hands.process(np.zeros((100, 100, 3), dtype=np.uint8))
        logger.info("[MEDIAPIPE] Strategy 3 OK")
        return hands
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Strategy 3 failed: {e}")

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        task_model = FRONTEND_DIR / "mp_models" / "hand_landmarker.task"
        if not task_model.exists():
            raise FileNotFoundError(f"Task model not found: {task_model}")
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        hands = mp_vision.HandLandmarker.create_from_options(options)
        hands._sign_turk_tasks_api = True
        hands._sign_turk_mp = mp
        logger.info("[MEDIAPIPE] Tasks API strategy OK")
        return hands
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Tasks API strategy failed: {e}")

    logger.error("[MEDIAPIPE] All strategies failed")
    return None


# ═══════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════
def load_model_assets():
    """Load the 179-class live recognition model and preprocessing assets."""
    global MODEL, LABEL_MAP, NORM_MEAN, NORM_STD, DEMO_CONFIG, MP_HANDS
    global SEQ_LEN, FEAT_DIM, NUM_CLASSES, CONFIDENCE_THRESHOLD, MODEL_LOAD_TIME

    t0 = time.time()

    cfg_path = MODEL_DIR / "demo_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            DEMO_CONFIG = json.load(f)
        SEQ_LEN              = DEMO_CONFIG.get("seq_len", 16)
        FEAT_DIM             = DEMO_CONFIG.get("feat_dim", 156)
        NUM_CLASSES          = DEMO_CONFIG.get("num_classes", 179)
        CONFIDENCE_THRESHOLD = DEMO_CONFIG.get("confidence_threshold", 0.40)
        logger.info(f"[CONFIG] seq_len={SEQ_LEN} feat_dim={FEAT_DIM} num_classes={NUM_CLASSES}")

    lm_path = MODEL_DIR / "label_map.json"
    if lm_path.exists():
        with open(lm_path, encoding="utf-8") as f:
            LABEL_MAP = json.load(f)
        logger.info(f"[LABELS] Loaded {len(LABEL_MAP)} classes")

    ns_path = MODEL_DIR / "norm_stats.json"
    if ns_path.exists():
        with open(ns_path) as f:
            ns = json.load(f)
        NORM_MEAN = np.array(ns["mean"], dtype=np.float32)
        NORM_STD  = np.where(np.array(ns["std"], dtype=np.float32) < 1e-6, 1.0,
                             np.array(ns["std"], dtype=np.float32))

    model_path = MODEL_DIR / "model.keras"
    if model_path.exists():
        try:
            @tf.keras.utils.register_keras_serializable()
            class ReduceSumAxis1(tf.keras.layers.Layer):
                def call(self, inputs):
                    return tf.reduce_sum(inputs, axis=1)
                def compute_output_shape(self, input_shape):
                    return (input_shape[0], input_shape[2])

            MODEL = tf.keras.models.load_model(str(model_path))
            MODEL.predict(np.zeros((1, SEQ_LEN, FEAT_DIM), dtype=np.float32), verbose=0)
            logger.info(f"[MODEL-NEXUS] Yüklendi — {MODEL.input_shape} → {MODEL.output_shape}")
        except Exception as e:
            logger.error(f"[MODEL-NEXUS] Yüklenemedi: {e}")

    MP_HANDS = init_mediapipe()
    MODEL_LOAD_TIME = time.time() - t0
    logger.info(f"[STARTUP] SignTurk live assets loaded in {MODEL_LOAD_TIME:.2f}s")


def load_animation_assets():
    """Load legacy compatibility assets used by the animation endpoints."""
    global ANIM_MODEL, ANIM_LABEL_MAP, ANIM_NORM_MEAN, ANIM_NORM_STD
    global ANIM_SEQ_LEN, ANIM_CONF_THRESH, ANIM_TOP_K, ANIM_SINGLE_DIM
    global LANDMARKS_DIR, LANDMARK_INDEX

    if not ASSETS_DIR.exists():
        logger.warning(f"[ANIM] model_assets klasörü bulunamadı: {ASSETS_DIR}")
        return

    config_file = ASSETS_DIR / "demo_config.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        ANIM_SEQ_LEN    = cfg.get("seq_len", 30)
        ANIM_CONF_THRESH = cfg.get("confidence_threshold", 0.5)
        ANIM_TOP_K       = cfg.get("top_k_display", 3)
        logger.info(f"[ANIM-CONFIG] seq_len={ANIM_SEQ_LEN}")

        model_file = ASSETS_DIR / cfg.get("model_file", "")
        if model_file.exists():
            try:
                ANIM_MODEL = tf.keras.models.load_model(str(model_file))
                logger.info(f"[ANIM-MODEL] Yüklendi — {ANIM_MODEL.input_shape}")
            except Exception as e:
                logger.error(f"[ANIM-MODEL] Yüklenemedi: {e}")

        norm_file = ASSETS_DIR / cfg.get("norm_stats_file", "")
        if norm_file.exists():
            with open(norm_file) as f:
                nd = json.load(f)
            ANIM_NORM_MEAN = np.array(nd["mean"], dtype=np.float32)
            ANIM_NORM_STD  = np.array(nd["std"],  dtype=np.float32)

        label_file = ASSETS_DIR / cfg.get("label_map_file", "")
        if label_file.exists():
            with open(label_file, encoding="utf-8") as f:
                raw = json.load(f)
            ANIM_LABEL_MAP = {int(k): v for k, v in raw.items()}
            logger.info(f"[ANIM-LABELS] Loaded {len(ANIM_LABEL_MAP)} classes")

    # Landmark dizinini bul
    for d in _candidate_landmark_dirs:
        if d.is_dir():
            LANDMARKS_DIR = d
            break

    if LANDMARKS_DIR:
        for fname in os.listdir(LANDMARKS_DIR):
            if fname.endswith(".json"):
                LANDMARK_INDEX[fname[:-5].lower()] = LANDMARKS_DIR / fname
        logger.info(f"[ANIM-LANDMARKS] Loaded {len(LANDMARK_INDEX)} words")
    else:
        logger.warning("[ANIM-LANDMARKS] Landmark klasörü bulunamadı")


# ═══════════════════════════════════════════════════════════
#  PREPROCESSING — live recognition
# ═══════════════════════════════════════════════════════════
def extract_landmarks_from_frame(frame: np.ndarray, mirror: bool = False) -> Optional[np.ndarray]:
    if MP_HANDS is None:
        return None
    try:
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        left_lm = np.zeros(63, dtype=np.float32)
        right_lm = np.zeros(63, dtype=np.float32)

        if getattr(MP_HANDS, "_sign_turk_tasks_api", False):
            mp = MP_HANDS._sign_turk_mp
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = MP_HANDS.detect(image)
            hand_pairs = zip(results.hand_landmarks or [], results.handedness or [])
            for landmarks, handedness in hand_pairs:
                label = handedness[0].category_name
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
                ).flatten()
                if label == "Left":
                    left_lm = coords
                else:
                    right_lm = coords
        else:
            results = MP_HANDS.process(rgb)
            hand_pairs = zip(
                results.multi_hand_landmarks or [], results.multi_handedness or []
            )
            for landmarks, handedness in hand_pairs:
                label = handedness.classification[0].label
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
                ).flatten()
                if label == "Left":
                    left_lm = coords
                else:
                    right_lm = coords
        return np.concatenate([left_lm, right_lm])
    except Exception as e:
        logger.error(f"[MEDIAPIPE] Frame error: {e}")
        return np.zeros(126, dtype=np.float32)


def hand_is_present(hand_vec: np.ndarray) -> bool:
    return bool(np.linalg.norm(hand_vec) > MIN_HAND_NORM)


def count_detected_hands(landmarks: np.ndarray) -> int:
    if landmarks is None or landmarks.shape != (126,):
        return 0
    return int(hand_is_present(landmarks[:63])) + int(hand_is_present(landmarks[63:]))


def normalize_hands_relative(X):
    result = X.copy()
    T = result.shape[0]
    for start in [0, 63]:
        hand = result[:, start:start+63].reshape(T, 21, 3)
        hand_rel = hand - hand[:, 0:1, :]
        scale = np.linalg.norm(hand_rel[:, 9, :], axis=-1, keepdims=True)[:, :, np.newaxis]
        scale = np.where(scale < 1e-6, 1.0, scale)
        result[:, start:start+63] = (hand_rel / scale).reshape(T, 63)
    return result


def compute_finger_angles(X):
    T = X.shape[0]
    chains = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    angles_all = []
    for hs in [0, 63]:
        hand = X[:, hs:hs+63].reshape(T, 21, 3)
        ha = np.zeros((T, 15), dtype=np.float32)
        idx = 0
        for chain in chains:
            for i in range(len(chain) - 1):
                a = hand[:, (0 if i == 0 else chain[i-1]), :]
                b = hand[:, chain[i], :]
                c = hand[:, chain[i+1], :]
                v1, v2 = a - b, c - b
                cos_a = np.sum(v1*v2, axis=-1) / (
                    np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
                ha[:, idx] = np.arccos(np.clip(cos_a, -1, 1))
                idx += 1
        angles_all.append(ha)
    return np.concatenate([X, np.concatenate(angles_all, axis=-1)], axis=-1)


def zscore_normalize(X):
    if NORM_MEAN is None or NORM_STD is None:
        return X
    return (X - NORM_MEAN) / NORM_STD


def preprocess_sequence(raw_landmarks):
    x = normalize_hands_relative(raw_landmarks)
    x = compute_finger_angles(x)
    x = zscore_normalize(x)
    return x[np.newaxis, :, :].astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  PREPROCESSING — signn 3D animasyon (color+depth)
# ═══════════════════════════════════════════════════════════
def build_anim_feature_vector(color_seq, depth_seq):
    color_norm = (color_seq - ANIM_NORM_MEAN[:ANIM_SINGLE_DIM]) / ANIM_NORM_STD[:ANIM_SINGLE_DIM]
    depth_norm = (depth_seq - ANIM_NORM_MEAN[ANIM_SINGLE_DIM:]) / ANIM_NORM_STD[ANIM_SINGLE_DIM:]
    return np.concatenate([color_norm, depth_norm], axis=-1)[np.newaxis, :, :]


# ═══════════════════════════════════════════════════════════
#  INFERENCE — live recognition
# ═══════════════════════════════════════════════════════════
def get_label(model_index: int) -> Dict[str, Any]:
    key = str(model_index)
    if key in LABEL_MAP:
        e = LABEL_MAP[key]
        return {"class_id": model_index,
                "label_tr": e.get("TR", f"class_{model_index}"),
                "label_en": e.get("EN", f"class_{model_index}")}
    return {"class_id": model_index,
            "label_tr": f"class_{model_index}",
            "label_en": f"class_{model_index}"}


def run_inference(input_tensor):
    global AVG_INFERENCE_MS, INFERENCE_COUNT
    if MODEL is None:
        return None
    t0 = time.time()
    proba = MODEL.predict(input_tensor, verbose=0)
    elapsed_ms = (time.time() - t0) * 1000
    INFERENCE_COUNT += 1
    AVG_INFERENCE_MS = (AVG_INFERENCE_MS * (INFERENCE_COUNT-1) + elapsed_ms) / INFERENCE_COUNT

    top_indices = np.argsort(proba[0])[::-1][:5]
    best_idx = int(top_indices[0])
    best_conf = float(proba[0][best_idx])
    best_lbl = get_label(best_idx)

    return {
        "class_id": best_lbl["class_id"],
        "label_tr": best_lbl["label_tr"],
        "label_en": best_lbl["label_en"],
        "confidence": best_conf,
        "animation_key": f"{best_lbl['label_en']}_sign",
        "category": "Sign",
        "above_threshold": best_conf >= CONFIDENCE_THRESHOLD,
        "top_predictions": [
            {"class_id": int(i), **get_label(int(i)),
             "confidence": round(float(proba[0][i]), 4)}
            for i in top_indices[:3]
        ],
    }


def run_best_inference(sequence: np.ndarray):
    return run_inference(preprocess_sequence(sequence))


# ═══════════════════════════════════════════════════════════
#  INFERENCE — signn 3D animasyon
# ═══════════════════════════════════════════════════════════
def run_anim_inference(feature_input):
    if ANIM_MODEL is None:
        return None
    t0 = time.perf_counter()
    preds = ANIM_MODEL.predict(feature_input, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    top_indices = np.argsort(preds)[::-1][:ANIM_TOP_K]
    top_results = []
    for idx in top_indices:
        info = ANIM_LABEL_MAP.get(int(idx), {"TR": "?", "EN": "?"})
        top_results.append({
            "class_id":   int(idx),
            "TR":         info.get("TR", "?"),
            "EN":         info.get("EN", "?"),
            "confidence": float(preds[idx])
        })
    status = "ok" if top_results[0]["confidence"] >= ANIM_CONF_THRESH else "low_confidence"
    return {"status": status, "top_k": top_results, "latency_ms": round(latency_ms, 1)}


# ═══════════════════════════════════════════════════════════
#  SESSION STATE — signn WebSocket için
# ═══════════════════════════════════════════════════════════
class SessionState:
    def __init__(self):
        self.color_buffer = []
        self.depth_buffer = []
        self.collecting   = False

    def reset(self):
        self.color_buffer = []
        self.depth_buffer = []
        self.collecting   = False

    def is_ready(self):
        return len(self.color_buffer) >= ANIM_SEQ_LEN


# ═══════════════════════════════════════════════════════════
#  FRAME BUFFER — live WebSocket
# ═══════════════════════════════════════════════════════════
class FrameBuffer:
    def __init__(self, seq_len=16):
        self.seq_len = seq_len
        self.buffer: List[np.ndarray] = []

    def clear(self):
        self.buffer.clear()

    def keep_tail(self, n):
        if n <= 0:
            self.clear()
        elif len(self.buffer) > n:
            self.buffer = self.buffer[-n:]

    def add(self, landmarks):
        self.buffer.append(landmarks)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
        return np.stack(self.buffer, axis=0) if len(self.buffer) == self.seq_len else None

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════
#  DB HELPER
# ═══════════════════════════════════════════════════════════
def add_log_db(db: Session, level: str, message: str, user: str = "system"):
    log = models.Log(level=level, message=message, user=user)
    db.add(log)
    db.commit()


def decode_base64_image(data_url):
    try:
        encoded = data_url.split(",", 1)[-1] if "," in data_url else data_url
        arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"[DECODE] {e}")
        return None


# ── Schemas ───────────────────────────────────────────────
class SettingsData(BaseModel):
    camera: str
    voice: str
    speech_rate: float
    avatar_speed: float
    tts_enabled: bool = True
    notifications_enabled: bool = True
    avatar_enabled: bool = True
    websocket_enabled: bool = True

class RegisterData(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

class LoginData(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


# ═══════════════════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════════════════
app = FastAPI(title="SignTurk API", version="3.1.0")

_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

_bearer = HTTPBearer(auto_error=False)
_access_tokens: Dict[str, int] = {}


def _issue_access_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    _access_tokens[token] = user_id
    return token


def _user_for_token(token: str, db: Session):
    user_id = _access_tokens.get(token)
    if not user_id:
        return None
    return db.query(models.User).filter(models.User.id == user_id).first()


def current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
):
    user = _user_for_token(credentials.credentials, db) if credentials else None
    if not user or user.status != "Active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user


def admin_user(user=Depends(current_user)):
    if user.role != "Admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
        )
    return user

# ── Text-processing: sentence engine + rule-based grammar ───────────
# Deterministic rule-based corrector only. ML/LLM grammar and TTS stay
# OFF by default (GrammarCorrector.use_ml defaults to False, the live
# approval path never synthesizes audio). The optional /api/text/*
# routes expose the ML/TTS layer behind explicit opt-in flags.
try:
    from text_processing import GrammarCorrector
    SENTENCE_ENGINE = GrammarCorrector()  # use_ml=False by default
    logger.info("[TEXT] Sentence engine ready (rule-based; ML/LLM/TTS off)")
except Exception as e:
    SENTENCE_ENGINE = None
    logger.warning(f"[TEXT] Sentence engine unavailable: {e}")

try:
    from text_processing_routes import router as text_processing_router
    app.include_router(text_processing_router)
    logger.info("[TEXT] /api/text/* routes mounted")
except Exception as e:
    logger.warning(f"[TEXT] /api/text/* routes not mounted (optional deps?): {e}")


def assemble_sentence(words: List[str]) -> str:
    """Build a Turkish sentence from approved sign words (rule-based engine)."""
    cleaned = [w for w in (words or []) if w and str(w).strip()]
    if not cleaned:
        return ""
    if SENTENCE_ENGINE is None:
        return " ".join(cleaned)
    try:
        return SENTENCE_ENGINE.correct(cleaned)
    except Exception as e:
        logger.warning(f"[TEXT] sentence assembly failed: {e}")
        return " ".join(cleaned)


if FRONTEND_DIR.exists():
    # html=False: otomatik index.html sunmasın, "/" route'u biz yönetiyoruz
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="frontend")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ═══════════════════════════════════════════════════════════
#  ROUTES — temel
# ═══════════════════════════════════════════════════════════
@app.get("/")
async def root():
    for candidate in ["sign_turk_ui.html"]:
        f = FRONTEND_DIR / candidate
        if f.exists():
            return FileResponse(str(f))
    return JSONResponse(status_code=404, content={"error": "SignTurk UI not found"})


@app.get("/avatar3d")
async def avatar3d():
    """Senin 3D animasyon arayüzün."""
    # index.html, avatar3d.html olarak yeniden adlandırılmalı
    for candidate in ["avatar3d.html", "signn_index.html", "index.html"]:
        f = FRONTEND_DIR / candidate
        if f.exists():
            return FileResponse(str(f))
    return JSONResponse(status_code=404, content={"error": "3D arayuz bulunamadi"})

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "anim_model_loaded": ANIM_MODEL is not None,
        "mediapipe_ready": MP_HANDS is not None,
        "landmark_count": len(LANDMARK_INDEX),
        "database": database_name(),
        "seq_len": SEQ_LEN,
        "feat_dim": FEAT_DIM,
        "num_classes": NUM_CLASSES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ui_file": UI_FILE.name,
        "ui_ready": UI_FILE.exists(),
        "stream_paths": ["/api/predict/live", "/ws/stream"],
        "mediapipe_detection_threshold": MEDIAPIPE_DETECTION_CONFIDENCE,
        "mediapipe_tracking_threshold": MEDIAPIPE_TRACKING_CONFIDENCE,
        "mediapipe_static_image_mode": MEDIAPIPE_STATIC_IMAGE_MODE,
        "mediapipe_model_complexity": MEDIAPIPE_MODEL_COMPLEXITY,
        "missed_hand_grace_frames": MAX_MISSED_HAND_FRAMES,
        "live_buffer_stride": LIVE_BUFFER_STRIDE,
    }


# ═══════════════════════════════════════════════════════════
#  ROUTES — 3D animation
# ═══════════════════════════════════════════════════════════
@app.get("/signs")
async def list_signs():
    """Mevcut tüm işaret kelimelerini listele."""
    return {"words": sorted(LANDMARK_INDEX.keys()), "count": len(LANDMARK_INDEX)}

@app.get("/landmark/{word}")
async def get_landmark(word: str):
    """Kelimeye ait landmark verisini döndür (smooth edilmiş)."""
    key = word.lower().strip()
    path = LANDMARK_INDEX.get(key)
    if path is None:
        return Response(status_code=404, content=f"'{word}' bulunamadı")
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if SMOOTHER_AVAILABLE and not raw_data.get("smoothed"):
        raw_data = smooth_landmark_data(raw_data)
    return Response(
        content=json.dumps(raw_data, ensure_ascii=False),
        media_type="application/json"
    )

@app.websocket("/ws")
async def websocket_3d(websocket: WebSocket):
    """3D animasyon WebSocket — color+depth landmark akışı."""
    await websocket.accept()
    state = SessionState()
    logger.info("[WS-3D] Bağlantı kuruldu")
    try:
        async for message in websocket.iter_text():
            data     = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type == "frame":
                hand_detected = data.get("hand_detected", False)
                color_lm = np.array(data.get("color_landmarks", [0.0]*ANIM_SINGLE_DIM), dtype=np.float32)
                depth_lm = np.array(data.get("depth_landmarks", [0.0]*ANIM_SINGLE_DIM), dtype=np.float32)

                if not hand_detected:
                    if state.collecting:
                        state.reset()
                    await websocket.send_text(json.dumps({"type": "idle"}))
                    continue

                state.collecting = True
                state.color_buffer.append(color_lm)
                state.depth_buffer.append(depth_lm)

                await websocket.send_text(json.dumps({
                    "type":     "collecting",
                    "progress": round(len(state.color_buffer) / ANIM_SEQ_LEN, 2),
                    "frames":   len(state.color_buffer)
                }))

                if state.is_ready():
                    color_seq = np.array(state.color_buffer[:ANIM_SEQ_LEN])
                    depth_seq = np.array(state.depth_buffer[:ANIM_SEQ_LEN])
                    if ANIM_NORM_MEAN is not None:
                        feature_input = build_anim_feature_vector(color_seq, depth_seq)
                        loop   = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, run_anim_inference, feature_input)
                        if result:
                            result["type"] = "prediction"
                            await websocket.send_text(json.dumps(result))
                    state.reset()
                continue

            if msg_type == "reset":
                state.reset()
                await websocket.send_text(json.dumps({"type": "reset_ack"}))

    except WebSocketDisconnect:
        logger.info("[WS-3D] Bağlantı kesildi")
    except Exception as e:
        logger.error(f"[WS-3D] Hata: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════
#  ROUTES — Auth
# ═══════════════════════════════════════════════════════════
@app.post("/api/register")
@app.post("/api/auth/register")
def register(data: RegisterData, db: Session = Depends(get_db)):
    email = data.email.lower().strip()
    if db.query(models.User).filter(models.User.email == email).first():
        return JSONResponse(status_code=400, content={"error": "Email zaten kayıtlı"})
    user = models.User(
        full_name=data.full_name.strip(), email=email,
        password_hash=hash_pw(data.password),
        role="User", status="Active", sessions=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    add_log_db(db, "Info", f"Yeni kullanıcı: {data.email}")
    return {"message": "Kayıt başarılı",
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}

@app.post("/api/login")
@app.post("/api/auth/login")
def login(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == data.email.lower()).first()
    if not user or user.status != "Active" or not verify_pw(data.password, user.password_hash):
        return JSONResponse(status_code=401, content={"error": "Hatalı email veya şifre"})
    user.sessions += 1
    db.commit()
    add_log_db(db, "Success", f"Giriş: {user.email}")
    return {"message": "Giriş başarılı", "access_token": _issue_access_token(user.id),
            "token_type": "bearer",
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}


# ═══════════════════════════════════════════════════════════
#  ROUTES — live recognition WebSocket
# ═══════════════════════════════════════════════════════════
@app.websocket("/ws/stream")
@app.websocket("/api/predict/live")
async def live_predict(websocket: WebSocket):
    token = websocket.query_params.get("token", "")
    auth_db = next(get_db())
    try:
        authenticated_user = _user_for_token(token, auth_db)
    finally:
        auth_db.close()
    if not authenticated_user or authenticated_user.status != "Active":
        await websocket.close(code=1008, reason="Authentication required")
        return
    await websocket.accept()
    logger.info("[WS-LIVE] Bağlandı")
    db = next(get_db())
    add_log_db(db, "Info", "Live WebSocket bağlandı")
    buf = FrameBuffer(seq_len=SEQ_LEN)
    missed_hand_frames = 0
    # ── ✓/✗ approval state ─────────────────────────────────────────
    # Words are saved to History and added to the sentence strip ONLY
    # when the user confirms ("doğru"). last_candidate holds the most
    # recent above-threshold prediction awaiting that confirmation.
    session_id = uuid.uuid4().hex[:8]
    sentence_words: List[str] = []
    last_candidate: Optional[str] = None
    try:
        while True:
            incoming = await websocket.receive_text()
            landmarks = None
            image_payload = None
            mirror_frame = False
            try:
                payload = json.loads(incoming)
                action = payload.get("action") if isinstance(payload, dict) else None
                if action:
                    # ── ✓/✗ approval control channel ───────────────
                    if action == "approve":
                        word = (payload.get("word") or last_candidate or "").strip()
                        if word:
                            sentence_words.append(word)
                            # Persist ONLY on user confirmation ("doğru").
                            try:
                                db.add(models.History(
                                    session_id=session_id, mode="Live",
                                    result=word, confidence="approved",
                                ))
                                db.commit()
                            except Exception as e:
                                db.rollback()
                                logger.warning(f"[DB] approved word save skipped: {e}")
                        last_candidate = None
                        await websocket.send_json({
                            "type": "sentence_update",
                            "words": list(sentence_words),
                            "sentence": assemble_sentence(sentence_words),
                            "added": word or None,
                        })
                    elif action == "reject":
                        # User said "yanlış": drop the candidate, save nothing.
                        last_candidate = None
                        await websocket.send_json({"type": "rejected"})
                    elif action == "undo":
                        if sentence_words:
                            sentence_words.pop()
                        await websocket.send_json({
                            "type": "sentence_update",
                            "words": list(sentence_words),
                            "sentence": assemble_sentence(sentence_words),
                        })
                    elif action in ("reset_sentence", "clear_sentence"):
                        sentence_words = []
                        last_candidate = None
                        await websocket.send_json({
                            "type": "sentence_update", "words": [], "sentence": "",
                        })
                    continue
                if "landmarks" in payload:
                    arr = np.array(payload["landmarks"], dtype=np.float32)
                    if arr.shape == (126,):
                        landmarks = arr
                else:
                    image_payload = (
                        payload.get("image")
                        or payload.get("frame")
                        or payload.get("data_url")
                        or payload.get("dataUrl")
                    )
                    mirror_frame = bool(payload.get("mirror", False))
            except (json.JSONDecodeError, ValueError):
                pass

            if landmarks is None:
                frame = decode_base64_image(image_payload or incoming)
                if frame is None:
                    await websocket.send_json({"class_id": -1, "label_tr": "-",
                        "label_en": "-", "confidence": 0, "animation_key": "none"})
                    continue
                landmarks = extract_landmarks_from_frame(frame, mirror=mirror_frame)
                if landmarks is None:
                    await websocket.send_json({"class_id": -1,
                        "label_tr": "MediaPipe kullanılamıyor",
                        "label_en": "Install mediapipe",
                        "confidence": 0, "animation_key": "none"})
                    continue

            hands_detected = count_detected_hands(landmarks)
            if hands_detected == 0:
                missed_hand_frames += 1
                if missed_hand_frames >= MAX_MISSED_HAND_FRAMES:
                    buf.clear()
                await websocket.send_json({
                    "class_id": -1,
                    "label_tr": "Takip korunuyor" if len(buf) else "El algilanmadi",
                    "label_en": "Tracking hold" if len(buf) else "No hand detected",
                    "confidence": 0,
                    "animation_key": "none",
                    "category": "-",
                    "hands_detected": 0,
                    "buffer_length": len(buf),
                    "missed_hand_frames": missed_hand_frames,
                })
                continue

            missed_hand_frames = 0
            sequence = buf.add(landmarks)
            if sequence is None:
                await websocket.send_json({"class_id": -1,
                    "label_tr": f"Buffering ({len(buf)}/{SEQ_LEN})",
                    "label_en": "Collecting...",
                    "confidence": len(buf)/SEQ_LEN,
                    "animation_key": "none",
                    "category": "-",
                    "hands_detected": hands_detected,
                    "buffer_length": len(buf)})
                continue

            if MODEL is None:
                await websocket.send_json({"class_id": -1,
                    "label_tr": "Model yüklenmedi", "label_en": "Model not loaded",
                    "confidence": 0, "animation_key": "none"})
                continue

            prediction = run_best_inference(sequence)
            if prediction is None:
                continue

            result = {
                "class_id": prediction["class_id"],
                "label_tr": prediction["label_tr"] if prediction["above_threshold"]
                            else f"({prediction['label_tr']}?)",
                "label_en": prediction["label_en"],
                "confidence": prediction["confidence"],
                "animation_key": prediction["animation_key"],
                "category": prediction["category"],
                "top_predictions": prediction.get("top_predictions", []),
                "above_threshold": prediction["above_threshold"],
                "threshold": CONFIDENCE_THRESHOLD,
                "variant": "normal",
                "hands_detected": hands_detected,
                "buffer_length": len(buf),
            }

            # Track the latest confident prediction as the approval candidate.
            # Nothing is persisted here — a word is saved to History and added
            # to the sentence strip ONLY when the user confirms ("doğru") via
            # the {"action":"approve"} control message handled above.
            if prediction["above_threshold"]:
                last_candidate = prediction["label_tr"]
            result["pending_word"] = last_candidate

            await websocket.send_json(result)
            buf.keep_tail(LIVE_BUFFER_STRIDE)

    except WebSocketDisconnect:
        logger.info("[WS-LIVE] Bağlantı kesildi")
        add_log_db(db, "Info", "Live WebSocket kesildi")
    except Exception as e:
        logger.error(f"[WS-LIVE] Hata: {e}\n{traceback.format_exc()}")
        add_log_db(db, "Warning", f"WebSocket hatası: {e}")
    finally:
        db.close()


@app.post("/api/predict/sequence")
async def predict_sequence(payload: Dict[str, Any]):
    try:
        raw = np.array(payload["landmarks"], dtype=np.float32)
        if raw.shape != (SEQ_LEN, 126):
            return JSONResponse(status_code=400,
                content={"error": f"Beklenen ({SEQ_LEN}, 126), gelen {list(raw.shape)}"})
        result = run_best_inference(raw)
        if result is None:
            return JSONResponse(status_code=503, content={"error": "Model yüklenmedi"})
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════
#  ROUTES — History / Settings / Dictionary / Admin / Avatar
# ═══════════════════════════════════════════════════════════
@app.get("/api/history")
def get_history(db: Session = Depends(get_db), _user=Depends(current_user)):
    rows = db.query(models.History).order_by(
        models.History.created_at.desc()).limit(100).all()
    return [{"id": r.session_id, "mode": r.mode, "result": r.result,
             "conf": r.confidence,
             "time": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else ""}
            for r in rows]

@app.get("/api/settings")
def get_settings(db: Session = Depends(get_db), _user=Depends(current_user)):
    s = db.query(models.Setting).first()
    if not s:
        return {"camera": "Default Camera", "voice": "Female Voice",
                "speech_rate": 1.0, "avatar_speed": 1.0,
                "tts_enabled": True, "notifications_enabled": True,
                "avatar_enabled": True, "websocket_enabled": True}
    return {"camera": s.camera, "voice": s.voice,
            "speech_rate": s.speech_rate, "avatar_speed": s.avatar_speed,
            "tts_enabled": s.tts_enabled, "notifications_enabled": s.notifications_enabled,
            "avatar_enabled": s.avatar_enabled, "websocket_enabled": s.websocket_enabled}

@app.post("/api/settings")
def save_settings(data: SettingsData, db: Session = Depends(get_db), _user=Depends(current_user)):
    s = db.query(models.Setting).first()
    if not s:
        s = models.Setting(id=1)
        db.add(s)
    for field, val in data.dict().items():
        setattr(s, field, val)
    db.commit()
    add_log_db(db, "Info", "Ayarlar güncellendi", "admin")
    return {"message": "Ayarlar kaydedildi", "data": data.dict()}

@app.get("/api/dictionary")
async def get_dictionary():
    return [{"classId": int(k), "tr": v.get("TR", "?"), "en": v.get("EN", "?"),
             "animation": f"{v.get('EN','unknown')}_sign", "category": "Sign"}
            for k, v in sorted(LABEL_MAP.items(), key=lambda x: int(x[0]))]

@app.get("/api/dictionary/{class_id}")
async def get_dictionary_by_id(class_id: int):
    info = get_label(class_id)
    return {"classId": info["class_id"], "tr": info["label_tr"],
            "en": info["label_en"], "animation": f"{info['label_en']}_sign"}

@app.get("/api/avatar/{class_id}")
async def get_avatar(class_id: int):
    info = get_label(class_id)
    return {"class_id": info["class_id"], "label_tr": info["label_tr"],
            "label_en": info["label_en"], "animation_key": f"{info['label_en']}_sign"}

@app.get("/api/admin/overview")
def admin_overview(db: Session = Depends(get_db), _user=Depends(current_user)):
    rows = db.query(models.History).order_by(
        models.History.created_at.desc()).limit(100).all()
    confs = []
    for r in rows:
        try:
            confs.append(float(r.confidence.replace("%", "")))
        except:
            pass
    return {
        "total_users": db.query(models.User).count(),
        "active_users": db.query(models.User).filter(models.User.status == "Active").count(),
        "translations": db.query(models.History).count(),
        "avg_confidence": round(sum(confs)/len(confs), 1) if confs else 0.0,
    }

@app.get("/api/admin/users")
def admin_users(db: Session = Depends(get_db), _admin=Depends(admin_user)):
    return [{"name": u.full_name, "email": u.email,
             "role": u.role, "sessions": u.sessions, "status": u.status}
            for u in db.query(models.User).all()]

@app.get("/api/admin/logs")
def admin_logs(db: Session = Depends(get_db), _admin=Depends(admin_user)):
    return [{"level": l.level, "message": l.message, "user": l.user,
             "when": l.created_at.strftime("%Y-%m-%d %H:%M") if l.created_at else ""}
            for l in db.query(models.Log).order_by(
                models.Log.created_at.desc()).limit(100).all()]

@app.get("/api/admin/model")
async def admin_model(_user=Depends(current_user)):
    return {
        "model_loaded": MODEL is not None,
        "model_version": "BiLSTM + Temporal Attention · 16 × 156",
        "num_classes": NUM_CLASSES,
        "label_map_size": len(LABEL_MAP),
        "input_shape": str(MODEL.input_shape) if MODEL is not None else "(None, 16, 156)",
        "output_shape": str(MODEL.output_shape) if MODEL is not None else "(None, 179)",
        "live_model": "LOADED" if MODEL else "NOT LOADED",
        "legacy_model": "LOADED" if ANIM_MODEL else "NOT LOADED",
        "live_classes": NUM_CLASSES,
        "anim_classes": len(ANIM_LABEL_MAP),
        "landmark_words": len(LANDMARK_INDEX),
        "mediapipe_ready": MP_HANDS is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "avg_inference": f"{AVG_INFERENCE_MS:.0f} ms" if INFERENCE_COUNT else "Henüz yok",
        "database": database_name(),
    }

@app.get("/api/debug/model-check")
async def debug_model_check(_admin=Depends(admin_user)):
    return {
        "live_model_loaded": MODEL is not None,
        "anim_model_loaded": ANIM_MODEL is not None,
        "mediapipe_ready": MP_HANDS is not None,
        "label_map_entries": len(LABEL_MAP),
        "anim_label_entries": len(ANIM_LABEL_MAP),
        "landmark_words": len(LANDMARK_INDEX),
        "norm_stats_loaded": NORM_MEAN is not None,
        "anim_norm_loaded": ANIM_NORM_MEAN is not None,
        "smoother_available": SMOOTHER_AVAILABLE,
        "config": DEMO_CONFIG,
    }


# ═══════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] Tables ready via %s", database_name())

    db = next(get_db())
    try:
        admin_email = os.environ.get("SIGNTURK_ADMIN_EMAIL", "").strip().lower()
        admin_password = os.environ.get("SIGNTURK_ADMIN_PASSWORD", "")
        if admin_email and admin_password and not db.query(models.User).filter(
            models.User.email == admin_email
        ).first():
            if len(admin_password) < 8:
                raise RuntimeError("SIGNTURK_ADMIN_PASSWORD must be at least 8 characters")
            db.add(models.User(
                full_name=os.environ.get("SIGNTURK_ADMIN_NAME", "SignTurk Admin").strip(),
                email=admin_email,
                password_hash=hash_pw(admin_password),
                role="Admin",
                status="Active",
                sessions=0,
            ))
            db.commit()
            logger.info("[DB] Administrator account created from environment")
        if db.query(models.Setting).count() == 0:
            db.add(models.Setting(id=1))
            db.commit()
        add_log_db(db, "Info", "SignTurk backend started", "system")
    finally:
        db.close()

    load_model_assets()
    load_animation_assets()

    logger.info("=" * 60)
    logger.info("  SignTurk Backend v3.1 Ready")
    logger.info(f"  Live Model:   {'LOADED' if MODEL else 'MISSING'}")
    logger.info(f"  Legacy Model: {'LOADED' if ANIM_MODEL else 'MISSING'}")
    logger.info(f"  Landmarks:    {len(LANDMARK_INDEX)} words")
    logger.info(f"  MediaPipe:    {'READY' if MP_HANDS else 'MISSING'}")
    logger.info(f"  Database:     {database_name()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

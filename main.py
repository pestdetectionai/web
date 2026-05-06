from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dotenv import load_dotenv
import uuid
import cv2
import numpy as np
import os
import base64
import json
import firebase_admin
from firebase_admin import credentials, db
import cloudinary
import cloudinary.uploader


# =========================================================
# ENV
# =========================================================

load_dotenv()


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best.pt").strip()
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / MODEL_FILENAME))).resolve()

HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "underdogquality/yolo11s-pest-detection"
).strip()

HF_MODEL_FILE = os.getenv(
    "HF_MODEL_FILE",
    MODEL_FILENAME
).strip()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

AUTO_DOWNLOAD_MODEL = os.getenv(
    "AUTO_DOWNLOAD_MODEL",
    "true"
).strip().lower() in {"1", "true", "yes", "on"}

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
DEBUG_DIR = BASE_DIR / "debug"
WEB_DIR = BASE_DIR / "web"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

APP_PUBLIC_BASE_URL = os.getenv("APP_PUBLIC_BASE_URL", "").strip()

FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").strip()
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
FIREBASE_SERVICE_ACCOUNT_JSON_B64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON_B64", "").strip()
FIREBASE_LOGS_PATH = "/api/analyze/logs"

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "").strip()
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "").strip()
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "smart-pest-detection").strip()

YOLO_CONFIDENCE = 0.08
YOLO_IOU = 0.40
YOLO_IMAGE_SIZE = 1280

MAX_YOLO_BOX_AREA_RATIO = 0.10
LOW_CONF_LARGE_BOX_CONF = 0.20
LOW_CONF_LARGE_BOX_AREA_RATIO = 0.040

VISUAL_COUNTER_ENABLED = True

# This is the main duplicate rule:
# if the fallback pest is almost inside a YOLO pest, remove fallback duplicate.
UNKNOWN_OVERLAP_WITH_YOLO = 0.45

FINAL_NMS_IOU = 0.10

GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
BLACK = (0, 0, 0)


# =========================================================
# APP INIT
# =========================================================

app = FastAPI(
    title="Smart Pest Trap Detection API",
    description="YOLO pest identification + visual pest counter + Cloudinary storage + Firebase logs + static website.",
    version="11.0.0"
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")
app.mount("/debug", StaticFiles(directory=str(DEBUG_DIR)), name="debug")
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


# =========================================================
# GLOBALS
# =========================================================

model = None
firebase_ready = False
cloudinary_ready = False


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():
    load_model()
    init_firebase()
    init_cloudinary()


def ensure_model_available():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        print("====================================")
        print("[MODEL] Local model found")
        print(f"[MODEL] Path: {MODEL_PATH}")
        print("====================================")
        return

    if not AUTO_DOWNLOAD_MODEL:
        raise RuntimeError(
            f"Model not found: {MODEL_PATH}\n"
            "AUTO_DOWNLOAD_MODEL=false, so boot download is disabled."
        )

    print("====================================")
    print("[MODEL] Local model not found")
    print("[MODEL] Downloading model from Hugging Face...")
    print(f"[MODEL] Repo: {HF_MODEL_REPO}")
    print(f"[MODEL] File: {HF_MODEL_FILE}")
    print(f"[MODEL] Save to: {MODEL_PATH.parent}")
    print("====================================")

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILE,
            local_dir=str(MODEL_PATH.parent),
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )

        downloaded_path = Path(downloaded_path).resolve()

        if downloaded_path != MODEL_PATH and downloaded_path.exists():
            MODEL_PATH.write_bytes(downloaded_path.read_bytes())

        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded model is missing or empty: {MODEL_PATH}")

        print("====================================")
        print("[MODEL] Download complete")
        print(f"[MODEL] Path: {MODEL_PATH}")
        print("====================================")

    except Exception as e:
        raise RuntimeError(
            f"Model download failed.\n"
            f"Repo: {HF_MODEL_REPO}\n"
            f"File: {HF_MODEL_FILE}\n"
            f"Target: {MODEL_PATH}\n"
            f"Error: {e}"
        )


def load_model():
    global model

    ensure_model_available()

    print("====================================")
    print("[MODEL] Loading pest detection model")
    print(f"[MODEL] Path: {MODEL_PATH}")
    print("====================================")

    model = YOLO(str(MODEL_PATH))

    print("[MODEL] Loaded successfully")


def init_firebase():
    global firebase_ready

    if not FIREBASE_DATABASE_URL:
        print("[FIREBASE] Disabled: FIREBASE_DATABASE_URL is missing")
        firebase_ready = False
        return

    try:
        if firebase_admin._apps:
            firebase_ready = True
            print("[FIREBASE] Already initialized")
            return

        if FIREBASE_SERVICE_ACCOUNT_JSON_B64:
            decoded = base64.b64decode(FIREBASE_SERVICE_ACCOUNT_JSON_B64).decode("utf-8")
            service_account_info = json.loads(decoded)
            cred = credentials.Certificate(service_account_info)
            print("[FIREBASE] Using FIREBASE_SERVICE_ACCOUNT_JSON_B64")

        elif FIREBASE_SERVICE_ACCOUNT_PATH:
            service_account_path = Path(FIREBASE_SERVICE_ACCOUNT_PATH)

            if not service_account_path.is_absolute():
                service_account_path = BASE_DIR / service_account_path

            if not service_account_path.exists():
                print(f"[FIREBASE] Service account file not found: {service_account_path}")
                firebase_ready = False
                return

            cred = credentials.Certificate(str(service_account_path))
            print(f"[FIREBASE] Using service account file: {service_account_path}")

        else:
            print("[FIREBASE] Disabled: service account is missing")
            firebase_ready = False
            return

        firebase_admin.initialize_app(
            cred,
            {
                "databaseURL": FIREBASE_DATABASE_URL
            }
        )

        firebase_ready = True
        print("[FIREBASE] Initialized successfully")

    except Exception as e:
        firebase_ready = False
        print(f"[FIREBASE] Init failed: {e}")


def init_cloudinary():
    global cloudinary_ready

    if not CLOUDINARY_CLOUD_NAME or not CLOUDINARY_API_KEY or not CLOUDINARY_API_SECRET:
        cloudinary_ready = False
        print("[CLOUDINARY] Disabled: missing CLOUDINARY_CLOUD_NAME/API_KEY/API_SECRET")
        return

    try:
        cloudinary.config(
            cloud_name=CLOUDINARY_CLOUD_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET,
            secure=True
        )

        cloudinary_ready = True
        print("[CLOUDINARY] Initialized successfully")

    except Exception as e:
        cloudinary_ready = False
        print(f"[CLOUDINARY] Init failed: {e}")


# =========================================================
# BASIC HELPERS
# =========================================================

def now_dt():
    return datetime.now()


def now_string():
    return now_dt().strftime("%Y-%m-%d %H:%M")


def now_iso():
    return now_dt().isoformat(timespec="seconds")


def now_timestamp_ms():
    return int(now_dt().timestamp() * 1000)


def get_base_url(request: Request):
    if APP_PUBLIC_BASE_URL:
        return APP_PUBLIC_BASE_URL.rstrip("/")
    return str(request.base_url).rstrip("/")


def validate_image_file(file: UploadFile):
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    return ext


async def save_upload(file: UploadFile, ext: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    with open(save_path, "wb") as buffer:
        buffer.write(content)

    if not save_path.exists() or save_path.stat().st_size <= 0:
        raise HTTPException(
            status_code=500,
            detail=f"Upload save failed: {save_path}"
        )

    return save_path


def safe_label(label: str):
    return label.replace("_", " ").strip()


def get_box(det):
    b = det["box"]
    return [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box

    x1 = max(0, min(float(x1), width - 1))
    y1 = max(0, min(float(y1), height - 1))
    x2 = max(0, min(float(x2), width - 1))
    y2 = max(0, min(float(y2), height - 1))

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def expand_box(box, pad, width, height):
    x1, y1, x2, y2 = box
    return clamp_box(
        [x1 - pad, y1 - pad, x2 + pad, y2 + pad],
        width,
        height
    )


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = box_area(box_a) + box_area(box_b) - inter

    if union <= 0:
        return 0.0

    return inter / union


def overlap_ratio_small(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    smaller = min(box_area(box_a), box_area(box_b))

    if smaller <= 0:
        return 0.0

    return inter / smaller


def nms_detections(detections, iou_threshold=0.10, class_aware=False):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: float(d.get("confidence", 0)), reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)

        remaining = []

        for det in detections:
            overlap = iou(get_box(best), get_box(det))

            if class_aware:
                if best["type"] == det["type"] and overlap > iou_threshold:
                    continue
            else:
                if overlap > iou_threshold:
                    continue

            remaining.append(det)

        detections = remaining

    return kept


# =========================================================
# CONFIDENCE HELPERS
# =========================================================

def compute_avg_confidence(item: dict) -> float:
    detections = item.get("detections", []) or []
    valid = []

    for det in detections:
        try:
            conf = float(det.get("confidence", 0) or 0)
            if conf > 0:
                valid.append(conf)
        except Exception:
            pass

    if not valid:
        return 0.0

    return round(sum(valid) / len(valid), 4)


def compute_top_confidence(item: dict) -> float:
    detections = item.get("detections", []) or []
    valid = []

    for det in detections:
        try:
            conf = float(det.get("confidence", 0) or 0)
            if conf > 0:
                valid.append(conf)
        except Exception:
            pass

    if not valid:
        return 0.0

    return round(max(valid), 4)


# =========================================================
# CLOUDINARY HELPERS
# =========================================================

def upload_image_to_cloudinary(file_path: Path, folder_name: str, public_id_prefix: str):
    if not cloudinary_ready:
        print("[CLOUDINARY] Skipped: cloudinary_ready=False")
        return None

    if not file_path:
        print("[CLOUDINARY] Skipped: file_path is None")
        return None

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"[CLOUDINARY] Skipped: file does not exist: {file_path}")
        return None

    if file_path.stat().st_size <= 0:
        print(f"[CLOUDINARY] Skipped: file is empty: {file_path}")
        return None

    try:
        public_id = f"{public_id_prefix}_{file_path.stem}"

        result = cloudinary.uploader.upload(
            str(file_path.resolve()),
            folder=f"{CLOUDINARY_FOLDER}/{folder_name}",
            public_id=public_id,
            resource_type="image",
            overwrite=True
        )

        return {
            "secure_url": result.get("secure_url"),
            "url": result.get("url"),
            "public_id": result.get("public_id"),
            "asset_id": result.get("asset_id"),
            "format": result.get("format"),
            "bytes": result.get("bytes"),
            "width": result.get("width"),
            "height": result.get("height")
        }

    except Exception as e:
        print(f"[CLOUDINARY] Upload failed for {file_path}: {e}")
        return None


def upload_analysis_images_to_cloudinary(uploaded_path: Path, result_image_path: Path, debug_mask_path: Path | None):
    timestamp_folder = datetime.now().strftime("%Y/%m/%d")

    original_cloud = upload_image_to_cloudinary(
        uploaded_path,
        f"{timestamp_folder}/original",
        "original"
    )

    annotated_cloud = upload_image_to_cloudinary(
        result_image_path,
        f"{timestamp_folder}/annotated",
        "annotated"
    )

    debug_cloud = None

    if debug_mask_path:
        debug_cloud = upload_image_to_cloudinary(
            debug_mask_path,
            f"{timestamp_folder}/debug",
            "debug"
        )

    return original_cloud, annotated_cloud, debug_cloud


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def remove_colored_markup_if_present(image):
    """
    Removes red user marks and green previous YOLO boxes/labels if a marked image is re-uploaded.
    Clean original frames are still best.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([14, 255, 255])

    lower_red2 = np.array([165, 70, 70])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.bitwise_or(red_mask, green_mask)

    if cv2.countNonZero(mask) < 50:
        return image

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.5,
        tileGridSize=(8, 8)
    )

    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.8, blur, -0.8, 0)

    return sharp


def find_trap_floor_crop(image):
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:10]:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            image_area = w * h

            if area > image_area * 0.16 and cw > w * 0.25 and ch > h * 0.25:
                pad = 2
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + pad)

                return image[y1:y2, x1:x2].copy(), x1, y1

    x1 = int(w * 0.16)
    y1 = int(h * 0.24)
    x2 = int(w * 0.82)
    y2 = int(h * 0.82)

    return image[y1:y2, x1:x2].copy(), x1, y1


# =========================================================
# YOLO DETECTION
# =========================================================

def yolo_predict(image, offset_x=0, offset_y=0, source_name="image"):
    results = model.predict(
        source=image,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        verbose=False
    )

    detections = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = safe_label(names.get(cls_id, str(cls_id)))

        xyxy = box.xyxy[0].cpu().numpy().astype(float)
        x1, y1, x2, y2 = xyxy.tolist()

        detections.append({
            "type": label,
            "confidence": round(confidence, 4),
            "source": source_name,
            "box": {
                "x1": round(x1 + offset_x, 2),
                "y1": round(y1 + offset_y, 2),
                "x2": round(x2 + offset_x, 2),
                "y2": round(y2 + offset_y, 2)
            }
        })

    return detections


def yolo_tiled(image, offset_x=0, offset_y=0):
    detections = []

    h, w = image.shape[:2]

    tile_size = 448
    overlap = 220
    step = tile_size - overlap

    y_positions = list(range(0, max(1, h - tile_size + 1), step))
    x_positions = list(range(0, max(1, w - tile_size + 1), step))

    if not y_positions:
        y_positions = [0]

    if not x_positions:
        x_positions = [0]

    last_y = max(0, h - tile_size)
    last_x = max(0, w - tile_size)

    if y_positions[-1] != last_y:
        y_positions.append(last_y)

    if x_positions[-1] != last_x:
        x_positions.append(last_x)

    for y in y_positions:
        for x in x_positions:
            tile = image[y:y + tile_size, x:x + tile_size].copy()

            tile_detections = yolo_predict(
                tile,
                offset_x=offset_x + x,
                offset_y=offset_y + y,
                source_name="tile"
            )

            detections.extend(tile_detections)

    return detections


def filter_bad_yolo_boxes(detections, image_width, image_height):
    filtered = []
    image_area = image_width * image_height

    for det in detections:
        box = get_box(det)
        area_ratio = box_area(box) / max(image_area, 1)
        conf = det["confidence"]

        bw = box[2] - box[0]
        bh = box[3] - box[1]

        if bw < 4 or bh < 4:
            continue

        if area_ratio > MAX_YOLO_BOX_AREA_RATIO:
            continue

        if conf < LOW_CONF_LARGE_BOX_CONF and area_ratio > LOW_CONF_LARGE_BOX_AREA_RATIO:
            continue

        filtered.append(det)

    return filtered


# =========================================================
# HARD VISUAL COUNTER
# =========================================================

def hard_visual_counter(original, floor_crop, floor_x, floor_y):
    """
    This is the important part.

    It counts visible dark pests using a hard visual threshold from the trap floor.
    For your sample image, it should find 4 dark components.
    YOLO will classify one, and this fallback will add the other 3 as unknown_pest.
    """
    detections = []

    crop_h, crop_w = floor_crop.shape[:2]
    original_h, original_w = original.shape[:2]

    gray = cv2.cvtColor(floor_crop, cv2.COLOR_BGR2GRAY)

    # Hard threshold is intentional.
    # Your missed pests are visibly dark. Dynamic threshold from the floor tends to include stains.
    # This threshold catches the 4 visible dark pest bodies/wings in the sample.
    threshold_value = int(os.getenv("VISUAL_DARK_THRESHOLD", "112"))

    mask = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    # Remove crop edge artifacts.
    border = 8
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    # Clean and merge wings/body.
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        contour_area = cv2.contourArea(cnt)
        box_area_local = bw * bh

        if box_area_local < 120:
            continue

        if box_area_local > 6000:
            continue

        if bw < 7 or bh < 7:
            continue

        aspect = bw / max(bh, 1)

        if aspect < 0.20 or aspect > 3.80:
            continue

        roi = gray[y:y + bh, x:x + bw]

        if roi.size == 0:
            continue

        contrast = float(np.std(roi))
        darkness = 255.0 - float(np.mean(roi))

        edges = cv2.Canny(roi, 10, 70)
        edge_density = cv2.countNonZero(edges) / max(1, bw * bh)

        # Real pests have strong darkness/contrast or wing/body edges.
        # Soft stains should fail this.
        if contrast < 18 and edge_density < 0.045:
            continue

        if darkness < 45 and contrast < 25:
            continue

        x1 = floor_x + x
        y1 = floor_y + y
        x2 = floor_x + x + bw
        y2 = floor_y + y + bh

        x1, y1, x2, y2 = expand_box(
            [x1, y1, x2, y2],
            pad=5,
            width=original_w,
            height=original_h
        )

        score = 0.18

        if contrast >= 25:
            score += 0.10

        if contrast >= 35:
            score += 0.10

        if edge_density >= 0.08:
            score += 0.08

        if darkness >= 90:
            score += 0.08

        if box_area_local >= 400:
            score += 0.04

        score = min(score, 0.60)

        detections.append({
            "type": "unknown_pest",
            "confidence": round(score, 4),
            "source": "hard_visual_counter",
            "debug": {
                "threshold": threshold_value,
                "local_x": int(x),
                "local_y": int(y),
                "local_w": int(bw),
                "local_h": int(bh),
                "box_area": int(box_area_local),
                "contour_area": round(float(contour_area), 2),
                "contrast": round(contrast, 2),
                "darkness": round(darkness, 2),
                "edge_density": round(edge_density, 4)
            },
            "box": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    detections = nms_detections(
        detections,
        iou_threshold=0.08,
        class_aware=False
    )

    return detections, mask


def remove_unknown_duplicates(yolo_detections, unknown_detections):
    """
    Remove unknown only if it is basically the same pest as a YOLO detection.
    """
    cleaned = []

    for unknown in unknown_detections:
        ub = get_box(unknown)
        duplicate = False

        for known in yolo_detections:
            kb = get_box(known)

            if overlap_ratio_small(ub, kb) >= UNKNOWN_OVERLAP_WITH_YOLO:
                duplicate = True
                break

        if not duplicate:
            cleaned.append(unknown)

    return cleaned


# =========================================================
# FULL DETECTION PIPELINE
# =========================================================

def run_detection_pipeline(image_path: Path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Uploaded image file does not exist before processing: {image_path}"
        )

    original = cv2.imread(str(image_path))

    if original is None:
        raise HTTPException(status_code=400, detail="Unable to read uploaded image.")

    original = remove_colored_markup_if_present(original)

    h, w = original.shape[:2]

    floor_crop, floor_x, floor_y = find_trap_floor_crop(original)
    enhanced_floor = enhance_image(floor_crop)

    yolo_detections = []

    yolo_detections.extend(
        yolo_predict(
            original,
            offset_x=0,
            offset_y=0,
            source_name="original"
        )
    )

    yolo_detections.extend(
        yolo_predict(
            floor_crop,
            offset_x=floor_x,
            offset_y=floor_y,
            source_name="trap_floor"
        )
    )

    yolo_detections.extend(
        yolo_predict(
            enhanced_floor,
            offset_x=floor_x,
            offset_y=floor_y,
            source_name="enhanced_floor"
        )
    )

    yolo_detections.extend(
        yolo_tiled(
            enhanced_floor,
            offset_x=floor_x,
            offset_y=floor_y
        )
    )

    fixed_yolo = []

    for det in yolo_detections:
        b = det["box"]

        x1, y1, x2, y2 = clamp_box(
            [b["x1"], b["y1"], b["x2"], b["y2"]],
            width=w,
            height=h
        )

        det["box"] = {
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2)
        }

        fixed_yolo.append(det)

    yolo_detections = filter_bad_yolo_boxes(
        fixed_yolo,
        image_width=w,
        image_height=h
    )

    yolo_detections = nms_detections(
        yolo_detections,
        iou_threshold=0.20,
        class_aware=True
    )

    visual_detections, visual_mask = hard_visual_counter(
        original,
        floor_crop,
        floor_x,
        floor_y
    )

    visual_detections = remove_unknown_duplicates(
        yolo_detections,
        visual_detections
    )

    final_detections = []
    final_detections.extend(yolo_detections)
    final_detections.extend(visual_detections)

    final_detections = nms_detections(
        final_detections,
        iou_threshold=FINAL_NMS_IOU,
        class_aware=False
    )

    return final_detections, original, visual_mask


# =========================================================
# DRAWING AND RESPONSE HELPERS
# =========================================================

def draw_annotated_image(original_image, image_path: Path, detections):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    image = original_image.copy()

    for det in detections:
        box = det["box"]
        label = det["type"]
        confidence = det["confidence"]

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        color = ORANGE if label == "unknown_pest" else GREEN
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        label_y1 = max(y1 - text_h - 10, 0)
        label_y2 = max(y1, text_h + 12)

        cv2.rectangle(
            image,
            (x1, label_y1),
            (min(x1 + text_w + 8, image.shape[1] - 1), label_y2),
            color,
            -1
        )

        cv2.putText(
            image,
            text,
            (x1 + 4, max(y1 - 6, text_h + 4)),
            font,
            font_scale,
            BLACK,
            thickness,
            cv2.LINE_AA
        )

    output_name = f"result_{image_path.stem}.jpg"
    output_path = RESULT_DIR / output_name

    success = cv2.imwrite(str(output_path), image)

    if not success or not output_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save annotated image: {output_path}"
        )

    return output_path


def save_debug_mask(image_path: Path, mask):
    if mask is None:
        return None

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    output_name = f"mask_{image_path.stem}.jpg"
    output_path = DEBUG_DIR / output_name

    success = cv2.imwrite(str(output_path), mask)

    if not success or not output_path.exists():
        print(f"[DEBUG] Failed to save debug mask: {output_path}")
        return None

    return output_path


def build_summary(detections):
    counts = Counter(det["type"] for det in detections)

    return [
        {
            "type": pest_type,
            "count": count
        }
        for pest_type, count in sorted(counts.items())
    ]


def get_local_image_urls(request: Request, uploaded_path: Path, result_image_path: Path, debug_mask_path: Path | None):
    base_url = get_base_url(request)

    original_image_url = f"{base_url}/uploads/{uploaded_path.name}"
    annotated_image_url = f"{base_url}/results/{result_image_path.name}"

    debug_mask_url = None
    if debug_mask_path is not None:
        debug_mask_url = f"{base_url}/debug/{debug_mask_path.name}"

    return original_image_url, annotated_image_url, debug_mask_url


# =========================================================
# FIREBASE LOG FUNCTIONS
# =========================================================

def firebase_logs_ref():
    return db.reference(FIREBASE_LOGS_PATH)


def save_analysis_log_to_firebase(log_payload: dict):
    if not firebase_ready:
        return None

    ref = firebase_logs_ref().push()
    log_id = ref.key

    log_payload["id"] = log_id
    log_payload["firebase_saved"] = True
    log_payload["firebase_path"] = f"{FIREBASE_LOGS_PATH}/{log_id}"

    ref.set(log_payload)

    return log_id


def get_all_logs_from_firebase():
    if not firebase_ready:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not initialized. Check FIREBASE_DATABASE_URL and service account."
        )

    raw = firebase_logs_ref().get()

    if not raw:
        return []

    logs = []

    for key, value in raw.items():
        if not isinstance(value, dict):
            continue

        item = value
        item["id"] = value.get("id", key)
        logs.append(item)

    return logs


def get_log_from_firebase(log_id: str):
    if not firebase_ready:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not initialized. Check FIREBASE_DATABASE_URL and service account."
        )

    item = firebase_logs_ref().child(log_id).get()

    if not item:
        raise HTTPException(status_code=404, detail="Log not found")

    item["id"] = item.get("id", log_id)
    return item


def parse_date_filter(value: str | None, end_of_day=False):
    if not value:
        return None

    try:
        if len(value) == 10:
            parsed = datetime.strptime(value, "%Y-%m-%d")
            if end_of_day:
                parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999000)
            return parsed

        return datetime.fromisoformat(value)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {value}. Use YYYY-MM-DD or ISO datetime."
        )


def filter_logs(
    logs,
    pest_type=None,
    date_from=None,
    date_to=None,
    min_total=None,
    max_total=None,
    search=None
):
    date_from_dt = parse_date_filter(date_from, end_of_day=False)
    date_to_dt = parse_date_filter(date_to, end_of_day=True)

    filtered = []

    for item in logs:
        total = int(item.get("total", 0) or 0)

        if min_total is not None and total < min_total:
            continue

        if max_total is not None and total > max_total:
            continue

        timestamp_ms = item.get("timestamp_ms")

        if timestamp_ms:
            item_dt = datetime.fromtimestamp(int(timestamp_ms) / 1000)
        else:
            item_dt = None

        if date_from_dt and item_dt and item_dt < date_from_dt:
            continue

        if date_to_dt and item_dt and item_dt > date_to_dt:
            continue

        data = item.get("data", [])
        detections = item.get("detections", [])

        if pest_type:
            wanted = pest_type.lower().strip()
            found_type = False

            for row in data:
                if str(row.get("type", "")).lower().strip() == wanted:
                    found_type = True
                    break

            for det in detections:
                if str(det.get("type", "")).lower().strip() == wanted:
                    found_type = True
                    break

            if not found_type:
                continue

        if search:
            s = search.lower().strip()
            haystack = json.dumps(item, ensure_ascii=False).lower()
            if s not in haystack:
                continue

        filtered.append(item)

    return filtered


def paginate_items(items, page, page_size):
    if page <= 0:
        page = 1

    if page_size <= 0:
        page_size = 10

    if page_size > 100:
        page_size = 100

    total_items = len(items)
    total_pages = max(1, (total_items + page_size - 1) // page_size)

    if page > total_pages:
        page_items = []
    else:
        start = (page - 1) * page_size
        end = start + page_size
        page_items = items[start:end]

    return {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
        "items": page_items
    }


def compact_log_item(item):
    return {
        "id": item.get("id"),
        "datatime": item.get("datatime"),
        "timestamp_ms": item.get("timestamp_ms"),
        "total": item.get("total", 0),
        "data": item.get("data", []),
        "avg_confidence": item.get("avg_confidence", compute_avg_confidence(item)),
        "top_confidence": item.get("top_confidence", compute_top_confidence(item)),
        "annotated_image": item.get("annotated_image"),
        "original_image": item.get("original_image"),
        "debug_mask": item.get("debug_mask"),
        "cloudinary": item.get("cloudinary", {})
    }


# =========================================================
# DASHBOARD HELPERS
# =========================================================

def build_dashboard_data(logs):
    logs = sorted(logs, key=lambda x: int(x.get("timestamp_ms", 0) or 0), reverse=True)

    today = datetime.now().date()
    seven_days_ago = datetime.now() - timedelta(days=6)

    total_logs = len(logs)
    total_pests = sum(int(item.get("total", 0) or 0) for item in logs)

    today_logs = []
    last_7_days_logs = []

    pest_counter = Counter()
    daily_counter = defaultdict(int)
    hourly_today_counter = defaultdict(int)

    for item in logs:
        timestamp_ms = item.get("timestamp_ms")

        if timestamp_ms:
            item_dt = datetime.fromtimestamp(int(timestamp_ms) / 1000)
        else:
            item_dt = None

        item_total = int(item.get("total", 0) or 0)

        for row in item.get("data", []):
            pest_counter[row.get("type", "unknown")] += int(row.get("count", 0) or 0)

        if item_dt:
            day_key = item_dt.strftime("%Y-%m-%d")
            daily_counter[day_key] += item_total

            if item_dt.date() == today:
                today_logs.append(item)
                hour_key = item_dt.strftime("%H:00")
                hourly_today_counter[hour_key] += item_total

            if item_dt >= seven_days_ago:
                last_7_days_logs.append(item)

    today_pests = sum(int(item.get("total", 0) or 0) for item in today_logs)

    top_pests = [
        {
            "type": pest_type,
            "count": count
        }
        for pest_type, count in pest_counter.most_common(10)
    ]

    daily_chart = []

    for i in range(6, -1, -1):
        day = datetime.now() - timedelta(days=i)
        key = day.strftime("%Y-%m-%d")
        daily_chart.append(
            {
                "date": key,
                "total": daily_counter.get(key, 0)
            }
        )

    hourly_chart = []

    for hour in range(24):
        key = f"{hour:02d}:00"
        hourly_chart.append(
            {
                "hour": key,
                "total": hourly_today_counter.get(key, 0)
            }
        )

    latest_log = logs[0] if logs else None

    recent_logs = [compact_log_item(item) for item in logs[:10]]

    return {
        "summary": {
            "total_logs": total_logs,
            "total_pests": total_pests,
            "today_logs": len(today_logs),
            "today_pests": today_pests,
            "last_7_days_logs": len(last_7_days_logs),
            "last_7_days_pests": sum(int(item.get("total", 0) or 0) for item in last_7_days_logs),
            "top_pests": top_pests
        },
        "chart": {
            "daily_last_7_days": daily_chart,
            "hourly_today": hourly_chart
        },
        "live_camera_stream": {
            "latest": compact_log_item(latest_log) if latest_log else None,
            "polling_route": "/api/live/latest",
            "note": "Use latest.annotated_image as the latest processed camera frame. Frontend can poll every 1 to 3 seconds."
        },
        "logs": recent_logs
    }


# =========================================================
# WEB UI ROUTES
# =========================================================

@app.get("/", include_in_schema=False)
def web_root():
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def ui_dashboard():
    index_path = WEB_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing web file: {index_path}"
        )

    return FileResponse(index_path)


@app.get("/ui/logs", include_in_schema=False)
def ui_logs():
    logs_path = WEB_DIR / "logs.html"

    if not logs_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing web file: {logs_path}"
        )

    return FileResponse(logs_path)


@app.get("/ui/logs/{log_id}", include_in_schema=False)
def ui_log_detail(log_id: str):
    detail_path = WEB_DIR / "detail.html"

    if not detail_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing web file: {detail_path}"
        )

    return FileResponse(detail_path)


@app.get("/ui/access", include_in_schema=False)
def ui_access():
    access_path = WEB_DIR / "access.html"

    if not access_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing web file: {access_path}"
        )

    return FileResponse(access_path)


# =========================================================
# API ROUTES
# =========================================================

@app.get("/api/status")
def api_status():
    return {
        "message": "Smart Pest Trap Detection API is running",
        "ui_route": "/ui",
        "analyze_route": "/api/analyze",
        "logs_route": "/api/logs",
        "log_detail_route": "/api/logs/{id}",
        "dashboard_route": "/api/dashboard",
        "live_latest_route": "/api/live/latest",
        "firebase_ready": firebase_ready,
        "cloudinary_ready": cloudinary_ready,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "auto_download_model": AUTO_DOWNLOAD_MODEL,
        "hf_model_repo": HF_MODEL_REPO,
        "hf_model_file": HF_MODEL_FILE,
        "firebase_logs_path": FIREBASE_LOGS_PATH,
        "cloudinary_folder": CLOUDINARY_FOLDER,
        "visual_dark_threshold": int(os.getenv("VISUAL_DARK_THRESHOLD", "112")),
        "field_name": "image"
    }


@app.post("/api/analyze")
async def analyze_pest(request: Request, image: UploadFile = File(...)):
    try:
        ext = validate_image_file(image)
        uploaded_path = await save_upload(image, ext)

        detections, processed_original, visual_mask = run_detection_pipeline(uploaded_path)

        result_image_path = draw_annotated_image(
            processed_original,
            uploaded_path,
            detections
        )

        debug_mask_path = save_debug_mask(
            uploaded_path,
            visual_mask
        )

        data = build_summary(detections)
        total = len(detections)

        avg_confidence = compute_avg_confidence({"detections": detections})
        top_confidence = compute_top_confidence({"detections": detections})

        local_original_url, local_annotated_url, local_debug_url = get_local_image_urls(
            request,
            uploaded_path,
            result_image_path,
            debug_mask_path
        )

        original_cloud, annotated_cloud, debug_cloud = upload_analysis_images_to_cloudinary(
            uploaded_path,
            result_image_path,
            debug_mask_path
        )

        original_image_url = original_cloud.get("secure_url") if original_cloud else local_original_url
        annotated_image_url = annotated_cloud.get("secure_url") if annotated_cloud else local_annotated_url
        debug_mask_url = debug_cloud.get("secure_url") if debug_cloud else local_debug_url

        cloudinary_saved = bool(original_cloud and annotated_cloud)

        log_payload = {
            "id": None,
            "datatime": now_string(),
            "created_at": now_iso(),
            "timestamp_ms": now_timestamp_ms(),
            "data": data,
            "total": total,
            "detections": detections,
            "avg_confidence": avg_confidence,
            "top_confidence": top_confidence,
            "original_image": original_image_url,
            "annotated_image": annotated_image_url,
            "debug_mask": debug_mask_url,
            "local_images": {
                "original_image": local_original_url,
                "annotated_image": local_annotated_url,
                "debug_mask": local_debug_url
            },
            "image_files": {
                "original_filename": uploaded_path.name,
                "annotated_filename": result_image_path.name,
                "debug_mask_filename": debug_mask_path.name if debug_mask_path else None
            },
            "cloudinary_saved": cloudinary_saved,
            "cloudinary": {
                "original": original_cloud,
                "annotated": annotated_cloud,
                "debug_mask": debug_cloud
            },
            "firebase_saved": False,
            "firebase_path": None,
            "note": "Green boxes are YOLO identified pests. Orange boxes are hard visual counter detections that YOLO could not classify."
        }

        log_id = save_analysis_log_to_firebase(log_payload)

        response = {
            "datatime": log_payload["datatime"],
            "id": log_id,
            "data": data,
            "total": total,
            "detections": detections,
            "avg_confidence": avg_confidence,
            "top_confidence": top_confidence,
            "original_image": original_image_url,
            "annotated_image": annotated_image_url,
            "debug_mask": debug_mask_url,
            "cloudinary_saved": cloudinary_saved,
            "firebase_saved": bool(log_id),
            "firebase_path": f"{FIREBASE_LOGS_PATH}/{log_id}" if log_id else None,
            "cloudinary": {
                "original": original_cloud,
                "annotated": annotated_cloud,
                "debug_mask": debug_cloud
            }
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/logs")
def list_logs(
    page: int = Query(1, description="Page number. If page=0, it becomes page=1."),
    page_size: int = Query(10, description="Items per page. Max 100."),
    pest_type: str | None = Query(None, description="Filter by pest type, example: unknown_pest"),
    date_from: str | None = Query(None, description="YYYY-MM-DD or ISO datetime"),
    date_to: str | None = Query(None, description="YYYY-MM-DD or ISO datetime"),
    min_total: int | None = Query(None),
    max_total: int | None = Query(None),
    search: str | None = Query(None),
    sort: str = Query("desc", description="desc or asc")
):
    logs = get_all_logs_from_firebase()

    logs = filter_logs(
        logs,
        pest_type=pest_type,
        date_from=date_from,
        date_to=date_to,
        min_total=min_total,
        max_total=max_total,
        search=search
    )

    reverse = sort.lower() != "asc"

    logs = sorted(
        logs,
        key=lambda x: int(x.get("timestamp_ms", 0) or 0),
        reverse=reverse
    )

    logs = [compact_log_item(item) for item in logs]

    result = paginate_items(logs, page, page_size)

    return {
        "success": True,
        "filters": {
            "pest_type": pest_type,
            "date_from": date_from,
            "date_to": date_to,
            "min_total": min_total,
            "max_total": max_total,
            "search": search,
            "sort": sort
        },
        **result
    }


@app.get("/api/logs/{log_id}")
def get_log_detail(log_id: str):
    item = get_log_from_firebase(log_id)

    return {
        "success": True,
        "data": item
    }


@app.get("/api/dashboard")
@app.get("/api/dashboard/")
def dashboard():
    logs = get_all_logs_from_firebase()
    dashboard_data = build_dashboard_data(logs)

    return {
        "success": True,
        "datatime": now_string(),
        **dashboard_data
    }


@app.get("/api/live/latest")
def live_latest():
    logs = get_all_logs_from_firebase()

    if not logs:
        return {
            "success": True,
            "latest": None
        }

    logs = sorted(
        logs,
        key=lambda x: int(x.get("timestamp_ms", 0) or 0),
        reverse=True
    )

    return {
        "success": True,
        "latest": compact_log_item(logs[0])
    }
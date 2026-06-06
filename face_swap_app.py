import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import io
import time
import uuid
import shutil
import logging
import base64
import zipfile
import traceback
import subprocess
import threading
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ──────────────────────────────────────────────
#  App bootstrap
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Model paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
INSWAPPER_MODEL_PATH = MODELS_DIR / "inswapper_128.onnx"

# ──────────────────────────────────────────────
#  Video processing storage & tracking
# ──────────────────────────────────────────────
TEMP_DIR = BASE_DIR / "temp_processing"
TEMP_DIR.mkdir(exist_ok=True)
# Global tracker for video swap tasks
VIDEO_TASKS = {}
BATCH_TASKS = {}

# ──────────────────────────────────────────────
#  InsightFace setup (lazy load)
# ──────────────────────────────────────────────
_face_analyser = None
_face_swapper = None

def get_execution_providers():
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
        logger.info(f"Available ONNX providers: {available}")
        if "CUDAExecutionProvider" in available:
            logger.info("NVIDIA GPU detected! Using CUDAExecutionProvider for acceleration.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception as e:
        logger.error(f"Error checking ONNX execution providers: {e}")
    logger.info("Using CPUExecutionProvider.")
    return ["CPUExecutionProvider"]

def get_face_analyser():
    global _face_analyser
    if _face_analyser is None:
        try:
            import insightface
            providers = get_execution_providers()
            _face_analyser = insightface.app.FaceAnalysis(
                name="buffalo_l",
                root=str(MODELS_DIR),
                providers=providers
            )
            _face_analyser.prepare(ctx_id=0, det_size=(320, 320))
            logger.info("Face analyser loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load face analyser: {e}")
            raise
    return _face_analyser

def get_face_swapper():
    global _face_swapper
    if _face_swapper is None:
        if not INSWAPPER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"INSwapper model not found at {INSWAPPER_MODEL_PATH}. "
                "Please wait for download_models.py to complete or download manually."
            )
        try:
            import insightface
            import onnxruntime
            
            sess_opts = onnxruntime.SessionOptions()
            sess_opts.intra_op_num_threads = 4
            sess_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            _face_swapper = insightface.model_zoo.get_model(
                str(INSWAPPER_MODEL_PATH),
                providers=get_execution_providers(),
                sess_options=sess_opts
            )
            logger.info("Face swapper loaded successfully with optimized session options.")
        except Exception as e:
            logger.error(f"Failed to load face swapper: {e}")
            raise
    return _face_swapper

# ──────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────
def decode_image(b64_data: str) -> np.ndarray:
    """Decode base64 image to OpenCV BGR array."""
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Make sure it is a valid JPG or PNG.")
    return img

def encode_image(img: np.ndarray, quality: int = 100, use_png: bool = False) -> str:
    """Encode OpenCV BGR image to base64 (lossless PNG or 100% quality JPEG)."""
    if use_png:
        _, buffer = cv2.imencode(".png", img)
        return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")
    else:
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def get_sorted_faces(faces: list, sort_by: str = "left-right"):
    """Sort faces by order preference."""
    if sort_by == "left-right":
        return sorted(faces, key=lambda f: f.bbox[0])
    elif sort_by == "right-left":
        return sorted(faces, key=lambda f: f.bbox[0], reverse=True)
    elif sort_by == "large-small":
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    elif sort_by == "small-large":
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return faces

def filter_by_gender(faces: list, gender_filter: str):
    """Filter faces by gender. gender attr: 0=female, 1=male."""
    if gender_filter == "male":
        return [f for f in faces if hasattr(f, 'gender') and f.gender == 1]
    elif gender_filter == "female":
        return [f for f in faces if hasattr(f, 'gender') and f.gender == 0]
    return faces

def blend_images(original: np.ndarray, swapped: np.ndarray, strength: float) -> np.ndarray:
    """Blend original and swapped images for strength control."""
    if strength >= 1.0:
        return swapped
    return cv2.addWeighted(original, 1.0 - strength, swapped, strength, 0)

def merge_audio(swapped_video_path: Path, original_video_path: Path, output_video_path: Path) -> bool:
    """Merge audio from original video into swapped video using ffmpeg if available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        # Check standard Windows paths as fallback
        std_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\tools\ffmpeg\bin\ffmpeg.exe",
        ]
        for p in std_paths:
            if os.path.exists(p):
                ffmpeg_path = p
                break
                
    if ffmpeg_path:
        try:
            logger.info("ffmpeg found. Attempting to copy audio...")
            # Command: ffmpeg -y -i swapped -i original -map 0:v -map 1:a? -c:v copy -c:a aac output
            cmd = [
                ffmpeg_path,
                "-y",
                "-i", str(swapped_video_path),
                "-i", str(original_video_path),
                "-map", "0:v",
                "-map", "1:a?",  # Optional audio map
                "-c:v", "copy",
                "-c:a", "aac",
                str(output_video_path)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            logger.info("ffmpeg audio merging completed successfully!")
            return True
        except Exception as e:
            logger.error(f"ffmpeg merging failed: {e}")
            
    # Fallback: copy swapped video to output as is (without audio)
    logger.warning("ffmpeg is not available or failed. Swapped video will be silent.")
    try:
        shutil.copy2(swapped_video_path, output_video_path)
    except Exception as e:
        logger.error(f"Fallback copy failed: {e}")
        return False
    return False

def process_video_task(task_id, source_img_path, target_video_path, strength, face_order, gender_filter, source_face_idx):
    """Process video frame-by-frame in a background thread."""
    swapped_temp_path = TEMP_DIR / f"{task_id}_temp.mp4"
    final_output_path = TEMP_DIR / f"{task_id}_swapped.mp4"
    
    try:
        # Mark as processing
        VIDEO_TASKS[task_id]["status"] = "processing"
        VIDEO_TASKS[task_id]["error"] = None
        
        source_img = cv2.imread(str(source_img_path))
        if source_img is None:
            raise ValueError("Could not read uploaded source image.")
            
        analyser = get_face_analyser()
        swapper = get_face_swapper()
        
        # Detect source face
        source_faces = analyser.get(source_img)
        if not source_faces:
            raise ValueError("No face detected in the source image. Please use a clear front-facing photo.")
            
        source_faces_sorted = get_sorted_faces(source_faces, face_order)
        if source_face_idx >= len(source_faces_sorted):
            source_face_idx = 0
        source_face = source_faces_sorted[source_face_idx]
        
        # Open video
        cap = cv2.VideoCapture(str(target_video_path))
        if not cap.isOpened():
            raise ValueError("Could not open uploaded target video.")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
            total_frames = total_frames or 100
            fps = fps or 30.0
            width = width or 640
            height = height or 480
            
        VIDEO_TASKS[task_id]["total_frames"] = total_frames
        
        # Define codec and writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(swapped_temp_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            original_frame = frame.copy()
            
            # Detect target faces on frame
            target_faces = analyser.get(frame)
            faces_swapped = 0
            
            if target_faces:
                target_faces_sorted = get_sorted_faces(target_faces, face_order)
                target_faces_filtered = filter_by_gender(target_faces_sorted, gender_filter)
                
                # Swap all detected/filtered faces
                for target_face in target_faces_filtered:
                    frame = swapper.get(frame, target_face, source_face, paste_back=True)
                    faces_swapped += 1
                    
                # Apply strength blend
                if strength < 1.0 and faces_swapped > 0:
                    frame = blend_images(original_frame, frame, strength)
                    
            out.write(frame)
            frame_idx += 1
            
            # Calculate speed and remaining time
            elapsed = time.time() - start_time
            avg_fps = frame_idx / elapsed if elapsed > 0 else 0
            rem_frames = total_frames - frame_idx
            eta_seconds = rem_frames / avg_fps if avg_fps > 0 else 0
            
            # Update progress status
            VIDEO_TASKS[task_id]["current_frame"] = frame_idx
            VIDEO_TASKS[task_id]["progress"] = round(frame_idx / total_frames, 3)
            VIDEO_TASKS[task_id]["fps"] = round(avg_fps, 1)
            VIDEO_TASKS[task_id]["eta"] = int(eta_seconds)
            
            # Generate preview every 15 frames (encode current frame to base64)
            if frame_idx % 15 == 1 or frame_idx == total_frames:
                try:
                    # Resize preview for fast transmission
                    preview_h = int(height * (240 / height)) if height > 0 else 240
                    preview_w = int(width * (240 / height)) if height > 0 else 320
                    preview_frame = cv2.resize(frame, (preview_w, preview_h))
                    _, buffer = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    b64_preview = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
                    VIDEO_TASKS[task_id]["preview"] = b64_preview
                except Exception:
                    pass
                
        # Release resources
        cap.release()
        out.release()
        
        # Merge audio
        VIDEO_TASKS[task_id]["status"] = "merging_audio"
        has_audio = merge_audio(swapped_temp_path, target_video_path, final_output_path)
        
        # Clean up files
        if swapped_temp_path.exists():
            try:
                swapped_temp_path.unlink()
            except Exception:
                pass
        if source_img_path.exists():
            try:
                source_img_path.unlink()
            except Exception:
                pass
        if target_video_path.exists():
            try:
                target_video_path.unlink()
            except Exception:
                pass
                
        # Mark as completed
        VIDEO_TASKS[task_id]["status"] = "completed"
        VIDEO_TASKS[task_id]["progress"] = 1.0
        VIDEO_TASKS[task_id]["result_file"] = str(final_output_path)
        VIDEO_TASKS[task_id]["has_audio"] = has_audio
        logger.info(f"Video swapping task {task_id} completed successfully.")
        
    except Exception as e:
        logger.exception(f"Error processing video task {task_id}")
        VIDEO_TASKS[task_id]["status"] = "failed"
        VIDEO_TASKS[task_id]["error"] = str(e)
        
        # Clean up files on error
        for p in [source_img_path, target_video_path, swapped_temp_path, final_output_path]:
            if p and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

def process_gif_task(task_id, source_img_path, target_gif_path, strength, face_order, gender_filter, source_face_idx):
    """Process a GIF frame-by-frame in a background thread."""
    final_output_path = TEMP_DIR / f"{task_id}_swapped.gif"
    try:
        VIDEO_TASKS[task_id]["status"] = "processing"
        VIDEO_TASKS[task_id]["error"] = None
        
        source_img = cv2.imread(str(source_img_path))
        if source_img is None:
            raise ValueError("Could not read uploaded source image.")
            
        analyser = get_face_analyser()
        swapper = get_face_swapper()
        
        # Detect source face
        source_faces = analyser.get(source_img)
        if not source_faces:
            raise ValueError("No face detected in the source image. Please use a clear portrait.")
            
        source_faces_sorted = get_sorted_faces(source_faces, face_order)
        if source_face_idx >= len(source_faces_sorted):
            source_face_idx = 0
        source_face = source_faces_sorted[source_face_idx]
        
        # Open target GIF using PIL
        gif = Image.open(str(target_gif_path))
        
        from PIL import ImageSequence
        total_frames = sum(1 for _ in ImageSequence.Iterator(gif))
        VIDEO_TASKS[task_id]["total_frames"] = total_frames
        
        # Get frame duration
        duration = gif.info.get('duration', 100)
        if duration <= 0:
            duration = 100
            
        swapped_frames = []
        
        # Process each frame
        for idx, frame in enumerate(ImageSequence.Iterator(gif)):
            # Convert frame to RGBA then BGR for OpenCV
            frame_rgba = frame.convert("RGBA")
            frame_np = np.array(frame_rgba)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
            frame_cv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
            original_frame = frame_cv.copy()
            
            # Detect target faces on frame
            target_faces = analyser.get(frame_cv)
            faces_swapped = 0
            
            if target_faces:
                target_faces_sorted = get_sorted_faces(target_faces, face_order)
                target_faces_filtered = filter_by_gender(target_faces_sorted, gender_filter)
                
                for target_face in target_faces_filtered:
                    frame_cv = swapper.get(frame_cv, target_face, source_face, paste_back=True)
                    faces_swapped += 1
                    
                if strength < 1.0 and faces_swapped > 0:
                    frame_cv = blend_images(original_frame, frame_cv, strength)
                    
            # Convert back to PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            swapped_frames.append(pil_frame)
            
            # Update progress
            VIDEO_TASKS[task_id]["current_frame"] = idx + 1
            VIDEO_TASKS[task_id]["progress"] = round((idx + 1) / total_frames, 3)
            
            # Generate preview base64 of completed result
            if idx % 10 == 0 or idx == total_frames - 1:
                try:
                    # Resize preview
                    h, w = frame_cv.shape[:2]
                    preview_h = int(h * (240 / h)) if h > 0 else 240
                    preview_w = int(w * (240 / h)) if h > 0 else 320
                    preview_frame = cv2.resize(frame_cv, (preview_w, preview_h))
                    _, buffer = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    b64_preview = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
                    VIDEO_TASKS[task_id]["preview"] = b64_preview
                except Exception:
                    pass
                    
        # Save compiled frames as animated GIF
        swapped_frames[0].save(
            str(final_output_path),
            save_all=True,
            append_images=swapped_frames[1:],
            duration=duration,
            loop=0
        )
        
        # Clean up temporary source and target files
        if source_img_path.exists():
            try: source_img_path.unlink()
            except Exception: pass
        if target_gif_path.exists():
            try: target_gif_path.unlink()
            except Exception: pass
            
        # Mark as completed
        VIDEO_TASKS[task_id]["status"] = "completed"
        VIDEO_TASKS[task_id]["progress"] = 1.0
        VIDEO_TASKS[task_id]["result_file"] = str(final_output_path)
        VIDEO_TASKS[task_id]["has_audio"] = False  # GIFs are always silent
        logger.info(f"GIF swapping task {task_id} completed successfully.")
        
    except Exception as e:
        logger.exception(f"Error processing GIF task {task_id}")
        VIDEO_TASKS[task_id]["status"] = "failed"
        VIDEO_TASKS[task_id]["error"] = str(e)
        
        # Clean up files on error
        for p in [source_img_path, target_gif_path, final_output_path]:
            if p and p.exists():
                try: p.unlink()
                except Exception: pass

def process_batch_task(batch_id, source_img_path, target_folder, results_folder, targets_metadata, strength, face_order, gender_filter, source_face_idx):
    """Process a batch of target images in a background thread."""
    try:
        BATCH_TASKS[batch_id]["status"] = "processing"
        
        # Load source face
        source_img = cv2.imread(str(source_img_path))
        if source_img is None:
            raise ValueError("Could not read uploaded source face image.")
            
        analyser = get_face_analyser()
        swapper = get_face_swapper()
        
        # Detect source face
        source_faces = analyser.get(source_img)
        if not source_faces:
            raise ValueError("No face detected in the source face image. Please use a clear portrait.")
            
        source_faces_sorted = get_sorted_faces(source_faces, face_order)
        if source_face_idx >= len(source_faces_sorted):
            source_face_idx = 0
        source_face = source_faces_sorted[source_face_idx]
        
        # Loop over target images
        for idx, item in enumerate(targets_metadata):
            target_filename = item["filename"]
            target_path = Path(item["path"])
            
            # Read target image
            target_img = cv2.imread(str(target_path))
            if target_img is None:
                # Skip invalid image
                logger.warning(f"Batch {batch_id}: Could not read target image {target_filename}. Skipping.")
                continue
                
            # Perform face swap
            target_faces = analyser.get(target_img)
            result_img = target_img.copy()
            swapped_count = 0
            
            if target_faces:
                target_faces_sorted = get_sorted_faces(target_faces, face_order)
                target_faces_filtered = filter_by_gender(target_faces_sorted, gender_filter)
                
                for target_face in target_faces_filtered:
                    result_img = swapper.get(result_img, target_face, source_face, paste_back=True)
                    swapped_count += 1
                    
                if strength < 1.0 and swapped_count > 0:
                    result_img = blend_images(target_img, result_img, strength)
            
            # Save swapped result as lossless PNG
            result_path = Path(results_folder) / f"swapped_{target_filename}"
            result_path = result_path.with_suffix(".png")
            
            cv2.imwrite(str(result_path), result_img)
            
            # Generate preview base64 of completed result
            preview_h = int(result_img.shape[0] * (240 / result_img.shape[0])) if result_img.shape[0] > 0 else 240
            preview_w = int(result_img.shape[1] * (240 / result_img.shape[0])) if result_img.shape[0] > 0 else 320
            preview_frame = cv2.resize(result_img, (preview_w, preview_h))
            _, buffer = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64_preview = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
            
            # Add to completed results
            BATCH_TASKS[batch_id]["completed_results"].append({
                "filename": result_path.name,
                "preview": b64_preview
            })
            
            # Update progress status
            BATCH_TASKS[batch_id]["completed"] = len(BATCH_TASKS[batch_id]["completed_results"])
            
        # All images processed! Create ZIP archive
        BATCH_TASKS[batch_id]["status"] = "zipping"
        
        zip_path = TEMP_DIR / f"{batch_id}_results.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for item in BATCH_TASKS[batch_id]["completed_results"]:
                filename = item["filename"]
                filepath = Path(results_folder) / filename
                if filepath.exists():
                    zipf.write(filepath, arcname=filename)
                    
        # Mark as completed
        BATCH_TASKS[batch_id]["status"] = "completed"
        BATCH_TASKS[batch_id]["zip_file"] = str(zip_path)
        logger.info(f"Batch swapping task {batch_id} completed successfully. Compiled ZIP: {zip_path}")
        
        # Clean up temporary source and target files
        if source_img_path.exists():
            try:
                source_img_path.unlink()
            except Exception:
                pass
        for item in targets_metadata:
            target_path = Path(item["path"])
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
                    
    except Exception as e:
        logger.exception(f"Error processing batch task {batch_id}")
        BATCH_TASKS[batch_id]["status"] = "failed"
        BATCH_TASKS[batch_id]["error"] = str(e)
        
        # Clean up on error
        if source_img_path.exists():
            try:
                source_img_path.unlink()
            except Exception:
                pass
        for item in targets_metadata:
            target_path = Path(item["path"])
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass

# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("faceswap.html")

@app.route("/api/models/status", methods=["GET"])
def models_status():
    """Return the status of required models."""
    inswapper_ready = INSWAPPER_MODEL_PATH.exists()
    analyser_ready = False
    try:
        get_face_analyser()
        analyser_ready = True
    except Exception:
        pass
    return jsonify({
        "inswapper": inswapper_ready,
        "analyser": analyser_ready,
        "inswapper_path": str(INSWAPPER_MODEL_PATH),
        "all_ready": inswapper_ready and analyser_ready,
    })

@app.route("/api/detect", methods=["POST"])
def detect_faces():
    """Detect faces in an image."""
    data = request.get_json(silent=True) or {}
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image provided."}), 400
    try:
        img = decode_image(img_b64)
        analyser = get_face_analyser()
        faces = analyser.get(img)
        result = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            gender = "male" if getattr(face, 'gender', 0) == 1 else "female"
            age = int(getattr(face, 'age', 0))
            det_score = float(getattr(face, 'det_score', 1.0))
            result.append({
                "bbox": bbox,
                "confidence": round(det_score, 3),
                "gender": gender,
                "age": age,
            })
        return jsonify({"faces": result, "count": len(result)})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Error during face detection")
        return jsonify({"error": f"Detection failed: {str(e)[:300]}"}), 500

@app.route("/api/swap", methods=["POST"])
def swap_faces():
    """Perform face swap on images."""
    data = request.get_json(silent=True) or {}
    source_b64 = data.get("source_image", "")
    target_b64 = data.get("target_image", "")
    source_face_index = int(data.get("source_face_index", 0))
    target_face_indices = data.get("target_face_indices", [])
    gender_filter = data.get("gender_filter", "all")
    face_order = data.get("face_order", "left-right")
    strength = max(0.0, min(1.0, float(data.get("strength", 1.0))))
    if not source_b64:
        return jsonify({"error": "No source image provided."}), 400
    if not target_b64:
        return jsonify({"error": "No target image provided."}), 400
    try:
        source_img = decode_image(source_b64)
        target_img = decode_image(target_b64)
        original_target = target_img.copy()
        analyser = get_face_analyser()
        swapper = get_face_swapper()
        source_faces = analyser.get(source_img)
        if not source_faces:
            return jsonify({"error": "No face detected in the source image. Please use a clear, front-facing photo."}), 422
        source_faces_sorted = get_sorted_faces(source_faces, face_order)
        if source_face_index >= len(source_faces_sorted):
            source_face_index = 0
        source_face = source_faces_sorted[source_face_index]
        target_faces = analyser.get(target_img)
        if not target_faces:
            return jsonify({"error": "No face detected in the target image."}), 422
        target_faces_sorted = get_sorted_faces(target_faces, face_order)
        target_faces_filtered = filter_by_gender(target_faces_sorted, gender_filter)
        if not target_faces_filtered:
            return jsonify({"error": f"No {gender_filter} faces found in target image after gender filter."}), 422
        if target_face_indices:
            selected_targets = []
            for idx in target_face_indices:
                if 0 <= idx < len(target_faces_filtered):
                    selected_targets.append(target_faces_filtered[idx])
            if not selected_targets:
                return jsonify({"error": "Specified face indices not found in target image."}), 422
        else:
            selected_targets = target_faces_filtered
        result_img = target_img.copy()
        faces_swapped = 0
        for target_face in selected_targets:
            result_img = swapper.get(result_img, target_face, source_face, paste_back=True)
            faces_swapped += 1
        if strength < 1.0:
            result_img = blend_images(original_target, result_img, strength)
        result_b64 = encode_image(result_img, use_png=True)
        return jsonify({
            "result_image": result_b64,
            "faces_swapped": faces_swapped,
            "source_faces_found": len(source_faces),
            "target_faces_found": len(target_faces),
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Error during face swap")
        return jsonify({"error": f"Swap failed: {str(e)[:400]}"}), 500

@app.route("/api/swap_webcam", methods=["POST"])
def swap_webcam():
    """Perform real-time webcam frame swap."""
    data = request.get_json(silent=True) or {}
    source_b64 = data.get("source_image", "")
    target_b64 = data.get("target_frame", "")
    strength = max(0.0, min(1.0, float(data.get("strength", 1.0))))
    face_order = data.get("face_order", "left-right")
    gender_filter = data.get("gender_filter", "all")
    source_face_idx = int(data.get("source_face_index", 0))
    
    if not source_b64:
        return jsonify({"error": "No source image provided."}), 400
    if not target_b64:
        return jsonify({"error": "No target frame provided."}), 400
        
    try:
        source_img = decode_image(source_b64)
        target_img = decode_image(target_b64)
        h, w = target_img.shape[:2]
        if w > 640 or h > 480:
            target_img = cv2.resize(target_img, (640, 480))
        original_target = target_img.copy()
        
        analyser = get_face_analyser()
        swapper = get_face_swapper()
        
        source_faces = analyser.get(source_img)
        if not source_faces:
            return jsonify({"error": "No face detected in source image."}), 422
            
        source_faces_sorted = get_sorted_faces(source_faces, face_order)
        if source_face_idx >= len(source_faces_sorted):
            source_face_idx = 0
        source_face = source_faces_sorted[source_face_idx]
        
        target_faces = analyser.get(target_img)
        result_img = target_img.copy()
        swapped_count = 0
        
        if target_faces:
            target_faces_sorted = get_sorted_faces(target_faces, face_order)
            target_faces_filtered = filter_by_gender(target_faces_sorted, gender_filter)
            
            for target_face in target_faces_filtered:
                result_img = swapper.get(result_img, target_face, source_face, paste_back=True)
                swapped_count += 1
                
            if strength < 1.0 and swapped_count > 0:
                result_img = blend_images(original_target, result_img, strength)
                
        result_b64 = encode_image(result_img, quality=80)
        return jsonify({
            "result_frame": result_b64,
            "swapped_count": swapped_count
        })
    except Exception as e:
        logger.error(f"Webcam swap error: {e}")
        return jsonify({"error": f"Webcam swap failed: {str(e)[:200]}"}), 500

@app.route("/api/swap_video", methods=["POST"])
def swap_video():
    """Start video swap background task."""
    if 'source_image' not in request.files or 'target_video' not in request.files:
        return jsonify({"error": "Missing source_image or target_video file."}), 400
        
    source_file = request.files['source_image']
    video_file = request.files['target_video']
    
    strength = max(0.0, min(1.0, float(request.form.get("strength", 1.0))))
    face_order = request.form.get("face_order", "left-right")
    gender_filter = request.form.get("gender_filter", "all")
    source_face_idx = int(request.form.get("source_face_index", 0))
    
    if source_file.filename == '' or video_file.filename == '':
        return jsonify({"error": "Empty filename."}), 400
        
    task_id = str(uuid.uuid4())
    
    # Save files to temp directory
    source_ext = os.path.splitext(source_file.filename)[1] or ".jpg"
    video_ext = os.path.splitext(video_file.filename)[1] or ".mp4"
    
    source_path = TEMP_DIR / f"{task_id}_source{source_ext}"
    video_path = TEMP_DIR / f"{task_id}_target{video_ext}"
    
    source_file.save(str(source_path))
    video_file.save(str(video_path))
    
    # Initialize task state
    VIDEO_TASKS[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "fps": 0,
        "eta": 0,
        "preview": None,
        "result_file": None,
        "error": None
    }
    
    # Start background processing thread
    target_task = process_gif_task if video_ext.lower() == ".gif" else process_video_task
    thread = threading.Thread(
        target=target_task,
        args=(task_id, source_path, video_path, strength, face_order, gender_filter, source_face_idx),
        daemon=True
    )
    thread.start()
    
    return jsonify({"task_id": task_id, "message": "Video swap started successfully."})

@app.route("/api/video_status/<task_id>", methods=["GET"])
def video_status(task_id):
    """Query progress status of a video swap task."""
    task = VIDEO_TASKS.get(task_id)
    if not task:
        return jsonify({"error": "Task not found."}), 404
    return jsonify(task)

@app.route("/api/video_download/<task_id>", methods=["GET"])
def video_download(task_id):
    """Download the completed swapped video."""
    task = VIDEO_TASKS.get(task_id)
    if not task or task.get("status") != "completed":
        return jsonify({"error": "Task not found or not completed."}), 404
        
    filepath = task.get("result_file")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Result file not found."}), 404
        
    mimetype = "image/gif" if filepath.lower().endswith(".gif") else "video/mp4"
    download_name = "faceswap_result.gif" if filepath.lower().endswith(".gif") else "faceswap_video_result.mp4"
    return send_file(
        filepath,
        mimetype=mimetype,
        as_attachment=True,
        download_name=download_name
    )

@app.route("/api/swap_batch", methods=["POST"])
def swap_batch():
    """Start batch face swap background task."""
    if 'source_image' not in request.files:
        return jsonify({"error": "Missing source_image file."}), 400
        
    target_files = request.files.getlist('target_images[]')
    if not target_files or len(target_files) == 0:
        return jsonify({"error": "No target images uploaded."}), 400
        
    if len(target_files) > 20:
        return jsonify({"error": "Batch swap limit exceeded. Max 20 images allowed."}), 400
        
    source_file = request.files['source_image']
    if source_file.filename == '':
        return jsonify({"error": "Empty source filename."}), 400
        
    strength = max(0.0, min(1.0, float(request.form.get("strength", 1.0))))
    face_order = request.form.get("face_order", "left-right")
    gender_filter = request.form.get("gender_filter", "all")
    source_face_idx = int(request.form.get("source_face_index", 0))
    
    batch_id = str(uuid.uuid4())
    
    # Establish batch-specific directories
    batch_source_dir = TEMP_DIR / f"{batch_id}_source"
    batch_targets_dir = TEMP_DIR / f"{batch_id}_targets"
    batch_results_dir = TEMP_DIR / f"{batch_id}_results"
    
    batch_source_dir.mkdir(exist_ok=True)
    batch_targets_dir.mkdir(exist_ok=True)
    batch_results_dir.mkdir(exist_ok=True)
    
    # Save source file
    source_ext = os.path.splitext(source_file.filename)[1] or ".jpg"
    source_path = batch_source_dir / f"source{source_ext}"
    source_file.save(str(source_path))
    
    # Save target files and create metadata list
    targets_metadata = []
    for f in target_files:
        if f.filename == '':
            continue
        safe_name = secure_filename(f.filename)
        target_path = batch_targets_dir / safe_name
        f.save(str(target_path))
        targets_metadata.append({
            "filename": safe_name,
            "path": str(target_path)
        })
        
    if not targets_metadata:
        return jsonify({"error": "No valid target files were saved."}), 400
        
    # Initialize batch task state
    BATCH_TASKS[batch_id] = {
        "status": "pending",
        "total": len(targets_metadata),
        "completed": 0,
        "completed_results": [],
        "zip_file": None,
        "error": None
    }
    
    # Start background batch swap thread
    thread = threading.Thread(
        target=process_batch_task,
        args=(batch_id, source_path, batch_targets_dir, batch_results_dir, targets_metadata, strength, face_order, gender_filter, source_face_idx),
        daemon=True
    )
    thread.start()
    
    return jsonify({"batch_id": batch_id, "message": "Batch swap started successfully."})

@app.route("/api/batch_status/<batch_id>", methods=["GET"])
def batch_status(batch_id):
    """Query progress status of a batch swap task."""
    task = BATCH_TASKS.get(batch_id)
    if not task:
        return jsonify({"error": "Batch task not found."}), 404
    return jsonify(task)

@app.route("/api/batch_download/<batch_id>", methods=["GET"])
def batch_download(batch_id):
    """Download the completed batch result ZIP file."""
    task = BATCH_TASKS.get(batch_id)
    if not task or task.get("status") != "completed":
        return jsonify({"error": "Batch task not completed yet."}), 404
        
    zip_path = task.get("zip_file")
    if not zip_path or not os.path.exists(zip_path):
        return jsonify({"error": "ZIP file not found."}), 404
        
    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name="faceswap_batch_results.zip"
    )

@app.route("/api/batch_download_individual/<batch_id>/<filename>", methods=["GET"])
def batch_download_individual(batch_id, filename):
    """Download an individual completed swapped image from a batch."""
    results_folder = TEMP_DIR / f"{batch_id}_results"
    filepath = results_folder / secure_filename(filename)
    
    if not filepath.exists():
        return jsonify({"error": "Swapped image not found."}), 404
        
    return send_file(
        str(filepath),
        mimetype="image/png",
        as_attachment=True,
        download_name=filename
    )

# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  FaceSwap Studio — Starting on http://localhost:5001")
    logger.info(f"  Models directory: {MODELS_DIR}")
    logger.info(f"  INSwapper model: {'FOUND ✓' if INSWAPPER_MODEL_PATH.exists() else 'MISSING — download in progress'}")
    logger.info("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5001, threaded=True)

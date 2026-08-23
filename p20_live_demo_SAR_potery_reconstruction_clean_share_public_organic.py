#!/usr/bin/env python3
"""



# 2026 08 22 : Now Tanishqa is part of sounder Bots FTC 23270    ## contact us for mentoring or assistance.







FLL Unearthed Dinos (Team 27820)
"Artifact" Vision Prototype for Raspberry Pi

This script supports: 
Minor enchangements to the code to make it run faster on PI 2025 10 09

Use cases:
- Live camera classification demo
- Capture frames to build a dataset
- Headless mode that saves annotated images

IMPORTANT:
This is a STEM/FLL prototype. Real archaeological organic identification
typically requires lab analysis and context.


Given to WA Innovators on 2026 01 27 : multiple teams in FIRST community.

Super set changes 2026 04 11

"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np

# Try OpenCV import
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Run: sudo apt install -y python3-opencv")
    sys.exit(1)

# Try PiCamera2 first (best for modern Raspberry Pi OS)
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

# Try TFLite runtime (lightweight)
TFLITE_AVAILABLE = False
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except Exception:
    TFLITE_AVAILABLE = False

# Fallback to full TensorFlow if present
if not TFLITE_AVAILABLE:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        TFLITE_AVAILABLE = True
    except Exception:
        TFLITE_AVAILABLE = False


# ----------------------------
# Heuristic classifier
# ----------------------------
def heuristic_organic_score(bgr_img):
    """
    Very simple baseline:
    - Many organic items in a demo have brown/green hues and irregular texture.
    - Many non-organic demo items appear gray/bright plastic-like with smoother texture.

    This is NOT scientifically valid; it’s a demo-friendly fallback.
    Returns score in [0, 1], higher => more likely organic (for demo purposes).
    """
    # Resize for speed
    img = cv2.resize(bgr_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Rough "earthy" hue ranges for brown/green
    # Green-ish: ~35-85 in OpenCV hue scale (0-179)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

    # Brown is tricky in HSV; approximate with lower V and moderate S in orange/red hues
    # This is intentionally loose for demo variety.
    brown_mask1 = cv2.inRange(hsv, (5, 50, 20), (25, 255, 200))
    brown_mask2 = cv2.inRange(hsv, (0, 40, 10), (10, 255, 160))
    brown_mask = cv2.bitwise_or(brown_mask1, brown_mask2)

    earthy_mask = cv2.bitwise_or(green_mask, brown_mask)

    earthy_ratio = float(np.count_nonzero(earthy_mask)) / earthy_mask.size

    # Texture estimate using Laplacian variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize texture to a soft range
    texture_score = min(lap_var / 500.0, 1.0)

    # Combine: earthy color + texture
    score = 0.65 * earthy_ratio + 0.35 * texture_score
    score = max(0.0, min(1.0, score))
    return score


def heuristic_predict(bgr_img):
    score = heuristic_organic_score(bgr_img)
    if score >= 0.5:
        return "organic", score
    else:
        return "non-organic", 1.0 - score


# ----------------------------
# TFLite classifier
# ----------------------------
class TFLiteBinaryClassifier:
    def __init__(self, model_path, labels_path=None):
        if not TFLITE_AVAILABLE:
            raise RuntimeError(
                "TFLite not available. Install with:\n"
                "  pip3 install tflite-runtime\n"
                "or install TensorFlow (heavier)."
            )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Infer input size
        in_shape = self.input_details[0]["shape"]
        # Typical shape: [1, height, width, channels]
        self.height = int(in_shape[1])
        self.width = int(in_shape[2])

        self.labels = None
        if labels_path and os.path.isfile(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]

    def preprocess(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)

        input_dtype = self.input_details[0]["dtype"]

        # Common patterns:
        # - float32 models expect 0..1
        # - uint8 models expect 0..255
        if input_dtype == np.float32:
            x = resized.astype(np.float32) / 255.0
        else:
            x = resized.astype(input_dtype)

        x = np.expand_dims(x, axis=0)
        return x

    def predict(self, bgr_img):
        x = self.preprocess(bgr_img)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Handle common binary outputs:
        # - shape (1,1) sigmoid => probability of class1
        # - shape (1,2) softmax => two class probs
        out = np.array(out)

        if out.size == 1:
            prob_class1 = float(out.reshape(-1)[0])
            prob_class1 = max(0.0, min(1.0, prob_class1))
            # Interpret class1 as "organic" by default
            organic_prob = prob_class1
            non_prob = 1.0 - organic_prob
            label = "organic" if organic_prob >= 0.5 else "non-organic"
            conf = organic_prob if label == "organic" else non_prob
            return label, conf, {"organic": organic_prob, "non-organic": non_prob}

        # Softmax-like two outputs
        flat = out.reshape(-1)
        if flat.size >= 2:
            # assume index 0 = non-organic, index 1 = organic unless labels override
            probs = flat[:2].astype(float)

            # If they aren't normalized, normalize
            s = float(np.sum(probs))
            if s > 0:
                probs = probs / s

            mapping = {0: "non-organic", 1: "organic"}
            if self.labels and len(self.labels) >= 2:
                # Use user labels order
                mapping = {0: self.labels[0], 1: self.labels[1]}

            p0, p1 = float(probs[0]), float(probs[1])
            label0 = mapping[0]
            label1 = mapping[1]

            prob_map = {label0: p0, label1: p1}

            # If labels are reversed, still pick the max
            label = label0 if p0 >= p1 else label1
            conf = max(p0, p1)

            # Provide a normalized organic/non-organic view if possible
            canonical = {}
            canonical["organic"] = prob_map.get("organic", p1 if label1 == "organic" else 0.0)
            canonical["non-organic"] = prob_map.get("non-organic", p0 if label0 == "non-organic" else 0.0)

            return label, conf, canonical

        # Fallback
        return "unknown", 0.0, {}


# ----------------------------
# Camera abstraction
# ----------------------------
class Camera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.picam2 = None
        self.cap = None

        if PICAMERA2_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(0.2)
            except Exception:
                self.picam2 = None

        if self.picam2 is None:
            # Fallback to OpenCV capture (USB cam or legacy setup)
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera. Check connection and permissions.")

    def read(self):
        if self.picam2 is not None:
            # Picamera2 returns RGB
            frame_rgb = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr

        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()


# ----------------------------
# Utility
# ----------------------------
def draw_overlay(frame, label, conf, mode):
    h, w = frame.shape[:2]
    text = f"{label}  ({conf*100:.1f}%)"
    sub = f"mode: {mode}"

    # Background rectangle for readability
    cv2.rectangle(frame, (10, 10), (10 + 360, 80), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, sub, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


def save_frame(frame, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{label}_{ts}.jpg"
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, frame)
    return path


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FLL Unearthed Organic vs Non-Organic Vision Prototype"
    )
    parser.add_argument("--model", type=str, default="",
                        help="Path to .tflite model (optional).")
    parser.add_argument("--labels", type=str, default="",
                        help="Path to labels.txt (optional).")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI; saves annotated frames periodically.")
    parser.add_argument("--save-dir", type=str, default="captures",
                        help="Directory for captured/annotated images.")
    parser.add_argument("--save-every", type=float, default=0.0,
                        help="In headless mode, save a frame every N seconds (0 disables).")
    parser.add_argument("--force-heuristic", action="store_true",
                        help="Ignore model even if provided.")
    args = parser.parse_args()

    # Set up classifier
    clf = None
    mode = "heuristic"

    if args.model and not args.force_heuristic:
        try:
            clf = TFLiteBinaryClassifier(args.model, args.labels or None)
            mode = "tflite"
        except Exception as e:
            print(f"Model load failed, falling back to heuristic. Reason: {e}")
            clf = None
            mode = "heuristic"

    # Camera
    cam = Camera(width=args.width, height=args.height, fps=args.fps)

    last_save = time.time()

    print("\nControls (GUI mode):")
    print("  c  -> capture image to dataset folder with predicted label")
    print("  o  -> force save as organic")
    print("  n  -> force save as non-organic")
    print("  h  -> toggle heuristic/ml (if model loaded)")
    print("  q  -> quit\n")

    use_heuristic = (mode == "heuristic")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Camera read failed.")
                time.sleep(0.05)
                continue

            # Predict
            if clf is not None and not use_heuristic:
                label, conf, _ = clf.predict(frame)
                if label not in ("organic", "non-organic"):
                    # If custom labels used or unknown, keep display safe
                    label = str(label)
            else:
                label, conf = heuristic_predict(frame)

            annotated = frame.copy()
            annotated = draw_overlay(annotated, label, conf, "heuristic" if use_heuristic else "tflite")

            # Headless behavior
            if args.headless:
                if args.save_every and (time.time() - last_save) >= args.save_every:
                    path = save_frame(annotated, args.save_dir, label.replace(" ", "_"))
                    print(f"Saved: {path}")
                    last_save = time.time()
                time.sleep(0.02)
                continue

            # GUI mode
            cv2.imshow("FLL Unearthed Organic Classifier", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                path = save_frame(frame, args.save_dir, label.replace(" ", "_"))
                print(f"Captured ({label}): {path}")

            elif key == ord('o'):
                path = save_frame(frame, os.path.join(args.save_dir, "organic"), "organic")
                print(f"Forced organic capture: {path}")

            elif key == ord('n'):
                path = save_frame(frame, os.path.join(args.save_dir, "non-organic"), "non-organic")
                print(f"Forced non-organic capture: {path}")

            elif key == ord('h'):
                if clf is not None:
                    use_heuristic = not use_heuristic
                    print("Switched to:", "heuristic" if use_heuristic else "tflite")
                else:
                    print("No model loaded. Heuristic only.")

    finally:
        cam.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()



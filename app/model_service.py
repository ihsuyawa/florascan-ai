import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Resolve directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # app/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)              # plant-ai-train/

def find_file(filename: str) -> str:
    candidates = [
        os.path.join(CURRENT_DIR, "models", filename),
        os.path.join(CURRENT_DIR, filename),
        os.path.join(PROJECT_ROOT, "models", filename),
        os.path.join(PROJECT_ROOT, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not locate '{filename}'. Searched locations:\n" + "\n".join(candidates)
    )

MODEL_PATH = find_file("plant_disease_model.tflite")
CLASSES_PATH = find_file("classes.txt")


class ModelService:
    def __init__(self):
        # Load labels
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f.readlines() if line.strip()]

        # Load TFLite Model
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image: Image.Image):
        input_data = self.preprocess_image(image)

        # Set input tensor and invoke
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get output probabilities
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        top_idx = int(np.argmax(output_data))
        raw_class = self.class_names[top_idx]
        confidence = float(output_data[top_idx]) * 100

        print(f"\n[DEBUG] Top Class: {raw_class} ({confidence:.2f}%)")
        top_3_indices = np.argsort(output_data)[-3:][::-1]
        print("[DEBUG] Top 3 Predictions:")
        for idx in top_3_indices:
            print(f"  - {self.class_names[idx]}: {output_data[idx]*100:.2f}%")

        clean_label = raw_class.replace("___", ": ").replace("_", " ")

        return {
            "raw_label": raw_class,
            "diagnosis": clean_label,
            "confidence": f"{confidence:.1f}%"
        }

# Instantiate singleton for import
model_service = ModelService()
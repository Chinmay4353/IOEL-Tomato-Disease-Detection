from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# MODEL PATHS
# ============================================================

# Model 1: Tomato Plant/Leaf Verification
GATE_MODEL = Path("models/model1_v2_stage2_best.keras")

# Model 2: Tomato Disease Classification
DISEASE_MODEL = Path("models/tomato_disease_finetuned.tflite")

CLASS_FILE = Path("models/tomato_disease_classes.json")


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (160, 160)

# Model 1 V2 selected threshold
GATE_THRESHOLD = 0.20

# Existing Disease Model 2 threshold
DISEASE_THRESHOLD = 0.80


# ============================================================
# LOAD MODEL 1 - V2
# ============================================================

print("Loading Model 1 V2...")

gate_model = tf.keras.models.load_model(
    GATE_MODEL,
    compile=False
)

print("Model 1 V2 loaded.")


# ============================================================
# LOAD MODEL 2 - TFLITE
# ============================================================

print("Loading Disease Model 2...")

disease_interpreter = tf.lite.Interpreter(
    model_path=str(DISEASE_MODEL)
)

disease_interpreter.allocate_tensors()

disease_input_details = disease_interpreter.get_input_details()
disease_output_details = disease_interpreter.get_output_details()

print("Disease Model 2 loaded.")


# ============================================================
# LOAD DISEASE CLASS NAMES
# ============================================================

with open(CLASS_FILE, "r", encoding="utf-8") as file:
    class_names = json.load(file)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image_path):
    """
    Preprocessing required by Model 1 V2:
    RGB -> 160x160 -> float32 -> normalize to 0-1
    """

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)

        array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    return np.expand_dims(array, axis=0)


# ============================================================
# DISEASE MODEL PREPROCESSING
# ============================================================

def preprocess_disease(image_path):
    """
    Existing preprocessing for Disease Model 2.
    """

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)

        array = np.asarray(
            image,
            dtype=np.float32
        )

    return np.expand_dims(array, axis=0)


# ============================================================
# RUN DISEASE MODEL
# ============================================================

def predict_disease(image_path):

    input_data = preprocess_disease(image_path)

    input_dtype = disease_input_details[0]["dtype"]

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32)

    else:
        input_scale, input_zero_point = disease_input_details[0]["quantization"]

        if input_scale > 0:
            input_data = (
                input_data / input_scale
                + input_zero_point
            ).astype(input_dtype)

    disease_interpreter.set_tensor(
        disease_input_details[0]["index"],
        input_data
    )

    disease_interpreter.invoke()

    output = disease_interpreter.get_tensor(
        disease_output_details[0]["index"]
    )

    output = np.asarray(output).flatten()

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])

    # Handle quantized output if required
    output_scale, output_zero_point = disease_output_details[0]["quantization"]

    if output_scale > 0 and disease_output_details[0]["dtype"] != np.float32:
        output = (
            output.astype(np.float32) - output_zero_point
        ) * output_scale

        predicted_index = int(np.argmax(output))
        confidence = float(output[predicted_index])

    prediction = class_names[predicted_index]

    return prediction, confidence


# ============================================================
# COMPLETE TWO-MODEL PIPELINE
# ============================================================

def predict(image_path):

    # --------------------------------------------------------
    # STEP 1: MODEL 1 V2 - TOMATO GATE
    # --------------------------------------------------------

    gate_input = preprocess(image_path)

    tomato_probability = float(
        gate_model.predict(
            gate_input,
            verbose=0
        )[0][0]
    )

    is_tomato = tomato_probability >= GATE_THRESHOLD

    # --------------------------------------------------------
    # STEP 2: REJECT NON-TOMATO
    # --------------------------------------------------------

    if not is_tomato:

        return {
            "is_tomato": False,
            "tomato_probability": tomato_probability,
            "prediction": "Not a tomato plant/leaf",
            "confidence": 1.0 - tomato_probability,
            "status": "NOT_TOMATO"
        }

    # --------------------------------------------------------
    # STEP 3: MODEL 2 - DISEASE CLASSIFIER
    # --------------------------------------------------------

    prediction, confidence = predict_disease(image_path)

    # --------------------------------------------------------
    # STEP 4: CONFIDENCE STATUS
    # --------------------------------------------------------

    if confidence >= DISEASE_THRESHOLD:
        status = "CONFIDENT"
    else:
        status = "UNCERTAIN"

    return {
        "is_tomato": True,
        "tomato_probability": tomato_probability,
        "prediction": prediction,
        "confidence": confidence,
        "status": status
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python src/inference/combined_inference.py <image_path>"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    result = predict(image_path)

    print()
    print("========== SmartStock / IOEL Inference ==========")
    print(f"is_tomato           : {result['is_tomato']}")
    print(
        f"tomato_probability  : "
        f"{result['tomato_probability']:.4f}"
    )
    print(f"prediction          : {result['prediction']}")
    print(
        f"confidence          : "
        f"{result['confidence']:.4f}"
    )
    print(f"status              : {result['status']}")
    print("==================================================")
    
class TomatoDiseasePipeline:

    def __init__(self):
        self.gate_model = gate_model
        self.disease_interpreter = disease_interpreter

    def predict(self, image_path):
        return predict(image_path)
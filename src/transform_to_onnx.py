import joblib
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

OUTPUT_DIR = Path("outputs")
LOCAL_MODEL_FILE = OUTPUT_DIR / "final_model.pkl"

def save_onnx():
    OUTPUT_DIR = Path("outputs")
    model = joblib.load(OUTPUT_DIR / "final_model.pkl")

    initial_type = [
        ("float_input", FloatTensorType([None, model.n_features_in_]))
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type
    )

    onnx_path = OUTPUT_DIR / "final_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ Export ONNX réussi dans {onnx_path}")
    
    
if __name__ == "__main__":
    save_onnx()
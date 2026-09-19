import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from prodml import config
from prodml.logging_conf import setup_logging

logger = setup_logging()


def export_to_onnx():
    logger.info("Loading pickle model for export...")
    with open(config.MODEL_PATH, "rb") as f:
        dv, model = pickle.load(f)

    num_features = len(dv.feature_names_)
    initial_type = [("float_input", FloatTensorType([None, num_features]))]

    logger.info("Converting to ONNX...")
    onx = convert_sklearn(model, initial_types=initial_type)

    onnx_path = config.MODEL_PATH.replace(".pkl", ".onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())

    logger.info(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    export_to_onnx()

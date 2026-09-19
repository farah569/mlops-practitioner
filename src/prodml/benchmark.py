import time
import pickle
import numpy as np
import onnxruntime as rt
from prodml import config
from prodml import data
from prodml import features


def run_benchmark():
    df = data.load_data()
    df = data.clean_data(df)
    df = features.create_features(df)
    df = df.head(500)

    with open(config.MODEL_PATH, "rb") as f:
        dv, pickle_model = pickle.load(f)

    train_dicts = features.prepare_dictionaries(
        df, config.CATEGORICAL, config.NUMERICAL
    )

    X = dv.transform(train_dicts).toarray().astype(np.float32)
    # test Pickle model
    start = time.time()
    pred_pkl = pickle_model.predict(X)
    end = time.time()
    pickle_latency = (end - start) * 1000

    # test ONNX model
    onnx_path = config.MODEL_PATH.replace(".pkl", ".onnx")
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    start = time.time()
    pred_onnx = sess.run(None, {input_name: X})[0].flatten()
    end = time.time()
    onnx_latency = (end - start) * 1000

    # Parity Test
    np.testing.assert_allclose(pred_pkl, pred_onnx, atol=1e-4)
    print("\n✅ Parity Test PASSED: Pickle and ONNX give the exact same predictions!")

    # print result
    print("\n📊 --- Latency Benchmark (500 rows) ---")
    print(f"Pickle Latency: {pickle_latency:.2f} ms")
    print(f"ONNX Latency:   {onnx_latency:.2f} ms")


if __name__ == "__main__":
    run_benchmark()

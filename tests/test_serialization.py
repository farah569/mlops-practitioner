import numpy as np
import onnxruntime as rt
import pickle
from prodml import config
from prodml import data
from prodml import features


def test_pickle_onnx_parity():
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

    pred_pkl = pickle_model.predict(X)

    onnx_path = config.MODEL_PATH.replace(".pkl", ".onnx")
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {input_name: X})[0].flatten()

    np.testing.assert_allclose(pred_pkl, pred_onnx, atol=1e-4)

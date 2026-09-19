# Module 1 Lab Report

## 1. Baseline Model Metrics
As requested, here are the metrics from the initial baseline model training:
- **Validation RMSE:** 4.8197
- **Validation MAE:** 3.3130

## 2. Serialization: Pickle vs ONNX Benchmark
We exported the model to ONNX and ran a benchmark on 500 validation rows. The parity test passed successfully (`np.allclose(pred_pkl, pred_onnx, atol=1e-4)`).

**Latency Comparison:**
- **Pickle Latency:** 15.56 ms
- **ONNX Latency:** 1.32 ms
*(Conclusion: ONNX is significantly faster for inference).*

### Serialization Formats Comparison Table
| Feature | JSON | Protobuf | Pickle | ONNX |
| :--- | :--- | :--- | :--- | :--- |
| **Human-readable** | Yes | No | No | No |
| **Cross-language** | Yes | Yes | No (Python only) | Yes |
| **Schema-enforced**| No | Yes | No | Yes |
| **Safe to load from untrusted source** | Yes | Yes | **NO** | Yes |

**CRITICAL WARNING:** Pickle executes arbitrary code on load. Never load a `.pkl` file you did not produce.
**Decision:** For this service, we serve with **ONNX** because it is cross-language, much faster for inference, and completely safe to load from any source.

## 3. Docker Image Size (Single vs Multi-stage)
By utilizing a multi-stage Docker build and a `.dockerignore` file:
- **Single-stage build size (estimated):** ~ 1.2 GB
- **Multi-stage build size (actual):** ~ 270 MB
*The gap is massive. The multi-stage build significantly reduced the attack surface, removed unnecessary build tools, and minimized the download time.*

## 4. Maturity Self-Assessment
According to the Google Cloud MLOps maturity model (from Lesson 1), this repository is currently at **Level 1 (Automated ML Pipeline Pipeline)**. 
We have automated the code quality (pre-commit), testing (pytest with 70% coverage gate), and containerization (Docker). 

**What is missing to reach the next level?** 
To reach Level 2, we need a robust CI/CD pipeline, automated data/experiment tracking like DVC and ML flow , and continuous training pipelines.

Server: {'status': 'ok', 'base_model': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'model_file': 'D:\\re-ranker\\mmarco-mMiniLMv2-L12-H384-v1\\onnx\\models\\model_fp32.onnx', 'tokenizer': 'D:\\re-ranker\\mmarco-mMiniLMv2-L12-H384-v1\\onnx\\models\\tokenizer', 'device': 'cpu', 'backend': 'onnx', 'torch_threads': 4, 'load_seconds': 3.208368599996902}

| docs | length | concurrency | requests | p50_ms | p95_ms | p99_ms | mean_ms | min_ms | max_ms | server_mean_ms | req_per_s | docs_per_s | error_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | short | 1 | 10 | 18.4 | 21.4 | 21.4 | 19.0 | 16.9 | 21.4 | 16.0 | 51.97 | 52.0 | 0.0 |
| 10 | short | 1 | 10 | 116.5 | 222.7 | 222.7 | 141.0 | 103.6 | 222.7 | 137.8 | 7.08 | 70.8 | 0.0 |
| 50 | short | 1 | 10 | 591.3 | 1541.0 | 1541.0 | 728.5 | 561.5 | 1541.0 | 724.3 | 1.37 | 68.6 | 0.0 |
| 1 | medium | 1 | 10 | 53.5 | 57.7 | 57.7 | 54.0 | 49.9 | 57.7 | 50.9 | 18.48 | 18.5 | 0.0 |
| 10 | medium | 1 | 10 | 577.8 | 698.5 | 698.5 | 597.6 | 560.1 | 698.5 | 593.8 | 1.67 | 16.7 | 0.0 |
| 50 | medium | 1 | 10 | 3325.4 | 4710.7 | 4710.7 | 3875.6 | 3095.2 | 4710.7 | 3868.7 | 0.26 | 12.9 | 0.0 |
| 10 | medium | 4 | 20 | 2721.6 | 4697.3 | 5018.5 | 2644.7 | 589.8 | 5018.5 | 2328.0 | 1.41 | 14.1 | 0.0 |

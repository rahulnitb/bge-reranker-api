Server: {'status': 'ok', 'base_model': 'Qwen/Qwen3-Reranker-0.6B', 'model_file': 'D:\\re-ranker\\Qwen3-Reranker-0.6B\\onnx\\models\\model_int8.onnx', 'tokenizer': 'D:\\re-ranker\\Qwen3-Reranker-0.6B\\onnx\\models\\tokenizer', 'device': 'cpu', 'backend': 'onnx-int8', 'torch_threads': 4, 'load_seconds': 4.472547699988354}

| docs | length | concurrency | requests | p50_ms | p95_ms | p99_ms | mean_ms | min_ms | max_ms | server_mean_ms | req_per_s | docs_per_s | error_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | short | 1 | 3 | 2536.4 | 2601.9 | 2601.9 | 2051.8 | 1017.0 | 2601.9 | 2041.8 | 0.48 | 0.5 | 0.0 |
| 10 | short | 1 | 3 | 10176.3 | 10194.5 | 10194.5 | 10143.5 | 10059.6 | 10194.5 | 10139.0 | 0.1 | 1.0 | 0.0 |
| 50 | short | 1 | 3 | 45745.0 | 45929.9 | 45929.9 | 45694.1 | 45407.3 | 45929.9 | 45684.9 | 0.02 | 1.1 | 0.0 |
| 1 | medium | 1 | 3 | 2553.8 | 2633.3 | 2633.3 | 2381.2 | 1956.6 | 2633.3 | 2377.3 | 0.42 | 0.4 | 0.0 |
| 10 | medium | 1 | 3 | 20484.3 | 20686.6 | 20686.6 | 20514.7 | 20373.2 | 20686.6 | 20509.8 | 0.05 | 0.5 | 0.0 |
| 50 | medium | 1 | 3 | 98691.8 | 101087.9 | 101087.9 | 99352.7 | 98278.3 | 101087.9 | 99345.4 | 0.01 | 0.5 | 0.0 |
| 10 | medium | 4 | 6 | 53528.8 | 69465.3 | 69465.3 | 54825.4 | 36655.8 | 69465.3 | 53641.9 | 0.06 | 0.6 | 0.0 |

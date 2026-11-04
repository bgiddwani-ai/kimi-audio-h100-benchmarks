# Results

The same three configurations were measured at concurrency 1, 2, 4, 8, 16, 32, 64 in two contexts. Each cell contains 320 requests. All 13,440 attempted requests completed successfully.

| Configuration, concurrency 8 | Original host req/s | Container req/s | Original p95 TTFT ms | Container p95 TTFT ms |
|---|---:|---:|---:|---:|
| Eager BF16 | 56.92 | 55.09 | 153.76 | 159.75 |
| Compile + CUDA graphs BF16 | 59.75 | 55.70 | 156.48 | 164.84 |
| TensorRT projector + vLLM BF16 | 58.55 | 52.62 | 157.98 | 173.39 |

- [Original host: all 21 cells](host/README.md)
- [Tested container workflow: all 21 cells](container/README.md)
- [Validation status](validation.json)

The original client ran in the host Python environment; the replication client runs inside the serving image. Each context has its own raw responses, summaries, logs, runtime provenance and graphs. Timing and batch-dependent answer differences are retained, not averaged across contexts. These short trials are observations, not production SLAs.

The Kubernetes manifest passed server dry-run validation. It was not scheduled as a live GPU pod because the source cluster has no NVIDIA DRA device inventory. The successful runtime replication used Docker.

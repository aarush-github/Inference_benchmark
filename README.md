# Inference Benchmark: vLLM vs SGLang on a single NVIDIA L4

A reproducible benchmark of two LLM serving engines, **vLLM 0.30.0** and **SGLang 0.5.20**, serving **Qwen3-8B** on a single 24 GB NVIDIA L4 (Lightning AI Studio, CUDA 13.0). It covers latency and throughput under load, FP8 weights and FP8 KV cache, prefix caching, structured output, prefill chunk size, and a GSM8K accuracy check.

**Read the write-up:** https://aarush-github.github.io/Inference_benchmark/

The short version: on this GPU the engine mattered less than the memory budget. The engines were level until KV-cache memory filled, and FP8 and prefix caching moved the numbers more than the choice of engine.

## Key results

| Finding | Result |
|---|---|
| Engines under light load | Throughput within about 4% up to 16 concurrent requests |
| p95 time to first token at 64 concurrent requests | 388 ms (vLLM BF16) vs 21.3 s (SGLang BF16), tracking KV-pool size (27,024 vs 12,505 tokens). Memory settings differed, so this is capacity, not an engine verdict |
| FP8 weights (vLLM) | Decode time per output token down 42.8% (67.25 to 38.47 ms) |
| FP8 KV cache at 64 concurrent requests | Throughput +19% on vLLM, +45% on SGLang; no effect up to 32 |
| Prefix caching (SGLang, 81% shared prompt) | p95 TTFT 13.4x lower (17.3 s to 1.3 s). Best case |
| JSON-constrained decoding (SGLang) | TPOT +21.6%, p99 inter-token latency 3.75x |
| GSM8K, all 1,319 questions, 5-shot | BF16 89.7%; FP8 settings 89.3% to 90.5%, within the plus or minus 1.6 point interval |

## What is in this repo

| Path | Contents |
|---|---|
| `src/` | Async load tester (`load_tested.py`, Python, asyncio, aiohttp) with streaming telemetry for TTFT, TPOT, ITL and VRAM (NVML) |
| `engines/` | Engine launch definitions for vLLM and SGLang |
| `configs/` | Run configurations |
| `scripts/` | Helper scripts for running benchmarks |
| `data/` | Raw JSON for every run, including `concurrency_sweep/` |
| `logs/` | Server logs (KV-pool sizes, in-flight requests, startup warnings) |
| `viz/` | Plotting and analysis code |
| `tests/` | Tests |
| `docs/` | The GitHub Pages write-up (`index.html`) and its chart images |

## Setup

Requirements: a CUDA GPU (results here are from a 24 GB NVIDIA L4), Python 3.12, and the two engines installed in separate environments.

```bash
# vLLM server (BF16 baseline)
vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.85 \
  --max-model-len 8192 --max-num-seqs 256

# SGLang server (BF16 baseline)
python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --mem-fraction-static 0.78 --disable-cuda-graph \
  --max-running-requests 256 --context-length 8192
```

FP8 runs: vLLM and SGLang used the `Qwen/Qwen3-8B-FP8` checkpoint for the speed benchmarks, and added `--kv-cache-dtype fp8` (vLLM) or `--kv-cache-dtype fp8_e4m3` (SGLang) for the FP8 KV-cache runs.

## Run a benchmark

Start one engine, then point the load tester at its OpenAI-compatible endpoint. The load tester calls `/v1/chat/completions` with `max_tokens` 256 in a closed loop at a fixed concurrency. Run `python src/load_tested.py --help` for its options.

The sweep ran concurrency 1 to 128 in doubling steps, 60 seconds per level.

## Regenerate the figures

```bash
python3 analysis/analyze.py      # writes summary.csv and every chart from the raw JSON
```

To redo the bandwidth chart with your own measurements instead of the spec and my estimates:

```bash
BW_GBS=<measured GB/s> WEIGHTS_BF16_GB=<n> WEIGHTS_W8A8_GB=<n> python3 analysis/analyze.py
```

## Accuracy check

Run with `lm-evaluation-harness` against a vLLM server, one configuration at a time:

```bash
lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3-8B,base_url=http://localhost:8000/v1/completions,tokenizer=Qwen/Qwen3-8B,num_concurrent=16,max_retries=3 \
  --tasks gsm8k --num_fewshot 5 \
  --gen_kwargs temperature=0,max_gen_toks=512 \
  --seed 0 --log_samples --output_path results/acc_bf16
```

The FP8 weight accuracy runs used vLLM's online dynamic FP8 on the BF16 model (`vllm serve Qwen/Qwen3-8B --quantization fp8`), not the pre-quantized `Qwen3-8B-FP8` checkpoint used in the speed benchmarks.

## Limitations

- **The 128-user sweep level is provisional.** My load tester capped in-flight requests near 100 (aiohttp's default connection limit), so that level did not reach 128 concurrent requests. Conclusions rest on 64 concurrent requests and below.
- **The engines ran with different memory settings** (vLLM `--gpu-memory-utilization 0.85`, SGLang `--mem-fraction-static 0.78`), and the two flags do not budget the same memory. Part of the KV-pool gap between engines is a settings gap.
- **SGLang ran without CUDA graphs in every run** (`--disable-cuda-graph`). Reruns with graphs enabled and a higher memory fraction are planned.
- **One run per setting**, except where noted. Run-to-run TTFT noise was about 30% at 10 concurrent requests.
- **One GPU, one dense 8B model, GSM8K as the only quality check.**
- An 11.9-second pause in the SGLang W8A8 run is unexplained.

The write-up lists every open item in detail.

## Citing

If you use this data, please link to this repository and the write-up.

## License

Add a license file (for example MIT) before sharing widely.

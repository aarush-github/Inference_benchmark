import asyncio
import aiohttp
import time
import json
import argparse
import numpy as np
from collections import Counter
from datetime import datetime, timezone
import pynvml


def init_nvml():
    """Initializes the NVML driver interface."""
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError as e:
        print(f"Warning: NVML initialization failed ({e}). VRAM metrics will be omitted.")
        return False


async def get_current_vram():
    """Jitter-free VRAM polling via native C API summing across visible GPUs."""
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        total_vram_mb = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_vram_mb += info.used // (1024 * 1024)
        return total_vram_mb
    except Exception:
        return None


async def sample_gpu_memory(interval, stop_event, samples):
    """Background task: poll NVML for GPU memory usage during the benchmark window."""
    while not stop_event.is_set():
        val = await get_current_vram()
        if val is not None:
            samples.append(val)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def send_stream_request(session, endpoint, model, prompt, error_counter, req_timeout=180):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start_time = time.time()
    first_token_time = None
    token_times = []
    usage = None

    timeout_cfg = aiohttp.ClientTimeout(total=req_timeout)

    try:
        async with session.post(endpoint, json=payload, timeout=timeout_cfg) as response:
            if response.status != 200:
                error_counter[f"http_{response.status}"] += 1
                return None
            async for line in response.content:
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                if decoded == "data: [DONE]":
                    break
                try:
                    chunk = json.loads(decoded[6:])
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        current_time = time.time()
                        if first_token_time is None:
                            first_token_time = current_time
                        token_times.append(current_time)

                if chunk.get("usage"):
                    usage = chunk["usage"]

    except asyncio.TimeoutError:
        error_counter["timeout"] += 1
        return None
    except aiohttp.ClientError as e:
        error_counter[f"client_error_{type(e).__name__}"] += 1
        return None
    except Exception as e:
        error_counter[f"other_{type(e).__name__}"] += 1
        return None

    end_time = time.time()
    if not token_times:
        error_counter["empty_response"] += 1
        return None

    ttft = first_token_time - start_time
    e2e = end_time - start_time
    last_token_time = token_times[-1]

    output_tokens = usage.get("completion_tokens", len(token_times)) if usage else len(token_times)
    input_tokens = usage.get("prompt_tokens") if usage else None

    # Precise TPOT: measured to arrival of final generated token
    tpot = (last_token_time - first_token_time) / (output_tokens - 1) if output_tokens > 1 else None
    itls = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    return {
        "ttft": ttft,
        "e2e": e2e,
        "tpot": tpot,
        "itls": itls,
        "output_tokens": output_tokens,
        "input_tokens": input_tokens,
    }


async def benchmark(endpoint, model, dataset_path, concurrency, num_requests, warmup, tag):
    with open(dataset_path, "r") as f:
        all_lines = [json.loads(line).get("text", "") for line in f.readlines()]

    total_required = warmup + num_requests
    if len(all_lines) < total_required:
        raise ValueError(
            f"Dataset too short! Requires {total_required} lines "
            f"({warmup} warmup + {num_requests} requests), but found only {len(all_lines)}."
        )

    warmup_prompts = all_lines[:warmup]
    prompts = all_lines[warmup:warmup + num_requests]

    error_counter = Counter()
    nvml_initialized = init_nvml()

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt):
            async with semaphore:
                return await send_stream_request(session, endpoint, model, prompt, error_counter)

        if warmup_prompts:
            print(f"Executing {len(warmup_prompts)} untimed warmup requests...")
            await asyncio.gather(*[bounded_request(p) for p in warmup_prompts])
            error_counter.clear()

        # Capture base VRAM before launching concurrent load
        base_vram = await get_current_vram() if nvml_initialized else None

        vram_samples = []
        stop_event = asyncio.Event()
        sampler_task = None
        if nvml_initialized:
            sampler_task = asyncio.create_task(sample_gpu_memory(0.5, stop_event, vram_samples))

        print(f"Benchmarking {len(prompts)} requests (concurrency={concurrency})...")
        tasks = [bounded_request(p) for p in prompts]
        start_benchmark = time.time()
        responses = await asyncio.gather(*tasks)
        end_benchmark = time.time()

        if nvml_initialized and sampler_task:
            stop_event.set()
            await sampler_task
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    results = [r for r in responses if r is not None]
    duration = end_benchmark - start_benchmark

    if not results:
        return {
            "tag": tag,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_requests": len(prompts),
            "successful_requests": 0,
            "duration_seconds": round(duration, 2),
            "errors": dict(error_counter),
        }

    total_output_tokens = sum(r["output_tokens"] for r in results)
    input_tokens_known = [r["input_tokens"] for r in results if r["input_tokens"] is not None]
    total_input_tokens = sum(input_tokens_known) if input_tokens_known else None

    all_ttft = [r["ttft"] * 1000 for r in results]
    all_tpot = [r["tpot"] * 1000 for r in results if r["tpot"] is not None]
    all_itls = [itl * 1000 for r in results for itl in r["itls"]]
    all_e2e = [r["e2e"] * 1000 for r in results]

    peak_vram = max(vram_samples) if vram_samples else None

    metrics = {
        "tag": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": model,
        "concurrency": concurrency,
        "total_requests": len(prompts),
        "successful_requests": len(results),
        "duration_seconds": round(duration, 2),
        "throughput_req_per_sec": round(len(results) / duration, 2),
        "output_tokens_per_sec": round(total_output_tokens / duration, 2),
        "input_tokens_per_sec": round(total_input_tokens / duration, 2) if total_input_tokens else None,
        "prompt_tokens_mean": round(float(np.mean(input_tokens_known)), 1) if input_tokens_known else None,
        "prompt_tokens_p50": int(np.percentile(input_tokens_known, 50)) if input_tokens_known else None,
        "prompt_tokens_p95": int(np.percentile(input_tokens_known, 95)) if input_tokens_known else None,
        "prompt_tokens_max": int(max(input_tokens_known)) if input_tokens_known else None,
        "ttft_p50_ms": round(np.percentile(all_ttft, 50), 2),
        "ttft_p95_ms": round(np.percentile(all_ttft, 95), 2),
        "ttft_p99_ms": round(np.percentile(all_ttft, 99), 2),
        "tpot_mean_ms": round(np.mean(all_tpot), 2),
        "tpot_p95_ms": round(np.percentile(all_tpot, 95), 2),
        "tpot_p99_ms": round(np.percentile(all_tpot, 99), 2),
        "itl_p95_ms": round(np.percentile(all_itls, 95), 2) if all_itls else 0,
        "itl_p99_ms": round(np.percentile(all_itls, 99), 2) if all_itls else 0,
        "latency_e2e_p50_ms": round(np.percentile(all_e2e, 50), 2),
        "latency_e2e_p95_ms": round(np.percentile(all_e2e, 95), 2),
        "latency_e2e_p99_ms": round(np.percentile(all_e2e, 99), 2),
        "vram_base_mb": base_vram,
        "vram_peak_mb": peak_vram,
        "vram_delta_mb": (peak_vram - base_vram) if (peak_vram and base_vram) else None,
        "vram_avg_mb": round(float(np.mean(vram_samples)), 1) if vram_samples else None,
        "vram_samples_collected": len(vram_samples),
        "errors": dict(error_counter),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5, help="Untimed requests run before measurement starts")
    parser.add_argument("--tag", type=str, default="", help="Run identifier, e.g. vllm_fp8_kvcache")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    final_metrics = asyncio.run(
        benchmark(args.endpoint, args.model, args.dataset, args.concurrency,
                  args.requests, args.warmup, args.tag)
    )

    with open(args.output, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(json.dumps(final_metrics, indent=2))
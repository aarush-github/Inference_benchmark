import asyncio
import aiohttp
import time
import json
import argparse
import numpy as np
from collections import Counter
from datetime import datetime, timezone

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


def init_nvml():
    """Initializes the NVML driver interface. Safe to call even if pynvml isn't installed."""
    if not _PYNVML_AVAILABLE:
        print("Warning: pynvml not installed. GPU metrics will be omitted.")
        return False
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError as e:
        print(f"Warning: NVML initialization failed ({e}). GPU metrics will be omitted.")
        return False


async def get_current_vram(gpu_index):
    """VRAM used (MB) on the single GPU this benchmark targets."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used // (1024 * 1024)
    except Exception:
        return None


async def get_gpu_utilization(gpu_index):
    """Utilization percentage for the single GPU this benchmark targets."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    except Exception:
        return None


async def sample_gpu_memory(gpu_index, interval, stop_event, samples, utilization_samples):
    """Poll GPU memory and utilization on one device during the benchmark window."""
    while not stop_event.is_set():
        val = await get_current_vram(gpu_index)
        if val is not None:
            samples.append(val)
        util = await get_gpu_utilization(gpu_index)
        if util is not None:
            utilization_samples.append(util)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def send_stream_request(session, endpoint, model, prompt, error_counter, system_prompt=None, response_format=None,
                               max_tokens=256, ignore_eos=False, req_timeout=180):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if response_format:
        payload["response_format"] = response_format

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

    if usage and isinstance(usage, dict):
        output_tokens = usage.get("completion_tokens", len(token_times))
        input_tokens = usage.get("prompt_tokens", None)
    else:
        output_tokens = len(token_times)
        input_tokens = None

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


def load_prompts_jsonl(path, total_required):
    """Reads prompts from a JSONL file, one {"text": ...} object per line.
    Standardized on JSONL across the project (not a single JSON array) so both
    load-testing scripts and the dataset format agree, and blank/corrupt trailing
    lines don't crash the whole run."""
    prompts = []
    with open(path, "r") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSONL line {line_num} in {path}")
                continue
            prompt_text = obj.get("prompt") or obj.get("text") or ""
            if prompt_text.strip():
                prompts.append(prompt_text)

    if len(prompts) < total_required:
        raise ValueError(
            f"Dataset too short! Requires {total_required} lines "
            f"but found only {len(prompts)} usable prompts in {path}."
        )
    return prompts


async def benchmark(endpoint, model, dataset_path, concurrency, num_requests, warmup, tag,
                     max_tokens, ignore_eos, gpu_index, system_prompt=None, response_format=None):
    total_required = warmup + num_requests
    all_lines = load_prompts_jsonl(dataset_path, total_required)

    warmup_prompts = all_lines[:warmup]
    prompts = all_lines[warmup:warmup + num_requests]

    error_counter = Counter()
    nvml_initialized = init_nvml()

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt):
            async with semaphore:
                return await send_stream_request(
                    session, endpoint, model, prompt, error_counter,
                    system_prompt=system_prompt, response_format=response_format,
                    max_tokens=max_tokens, ignore_eos=ignore_eos,
                )

        if warmup_prompts:
            print(f"Executing {len(warmup_prompts)} untimed warmup requests...")
            await asyncio.gather(*[bounded_request(p) for p in warmup_prompts])
            error_counter.clear()

        # Capture idle VRAM before launching concurrent load, so vram_delta_mb
        # reflects what the benchmark itself added, not what the server already held.
        base_vram = await get_current_vram(gpu_index) if nvml_initialized else None

        vram_samples = []
        gpu_util_samples = []
        stop_event = asyncio.Event()
        sampler_task = None
        if nvml_initialized:
            sampler_task = asyncio.create_task(
                sample_gpu_memory(gpu_index, 0.5, stop_event, vram_samples, gpu_util_samples)
            )

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

    peak_vram = max(vram_samples) if vram_samples else None
    avg_vram = round(float(np.mean(vram_samples)), 1) if vram_samples else None
    average_gpu_util = round(float(np.mean(gpu_util_samples)), 1) if gpu_util_samples else None
    maximum_gpu_util = max(gpu_util_samples) if gpu_util_samples else None
    vram_delta = (peak_vram - base_vram) if (peak_vram is not None and base_vram is not None) else None

    if not results:
        # Same schema as a successful run (all keys present, stats just null) so
        # downstream Week-6 aggregation never has to special-case a failed run.
        return {
            "tag": tag,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "model": model,
            "concurrency": concurrency,
            "total_requests": len(prompts),
            "successful_requests": 0,
            "duration_seconds": round(duration, 2),
            "throughput_req_per_sec": None,
            "output_tokens_per_sec": None,
            "input_tokens_per_sec": None,
            "prompt_tokens_mean": None,
            "prompt_tokens_p50": None,
            "prompt_tokens_p95": None,
            "prompt_tokens_max": None,
            "ttft_p50_ms": None,
            "ttft_p95_ms": None,
            "ttft_p99_ms": None,
            "tpot_mean_ms": None,
            "tpot_p95_ms": None,
            "tpot_p99_ms": None,
            "itl_p95_ms": None,
            "itl_p99_ms": None,
            "latency_e2e_p50_ms": None,
            "latency_e2e_p95_ms": None,
            "latency_e2e_p99_ms": None,
            "vram_base_mb": base_vram,
            "vram_peak_mb": peak_vram,
            "vram_delta_mb": vram_delta,
            "vram_avg_mb": avg_vram,
            "vram_samples_collected": len(vram_samples),
            "gpu_util_avg": average_gpu_util,
            "gpu_util_max": maximum_gpu_util,
            "errors": dict(error_counter),
        }

    total_output_tokens = sum(r["output_tokens"] for r in results)
    input_tokens_known = [r["input_tokens"] for r in results if r["input_tokens"] is not None]
    total_input_tokens = sum(input_tokens_known) if input_tokens_known else None

    all_ttft = [r["ttft"] * 1000 for r in results]
    all_tpot = [r["tpot"] * 1000 for r in results if r["tpot"] is not None]
    all_itls = [itl * 1000 for r in results for itl in r["itls"]]
    all_e2e = [r["e2e"] * 1000 for r in results]

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
        # tpot can legitimately be empty (every request produced <=1 output token,
        # e.g. a very small max_tokens) -- guard it the same way itl already was.
        "tpot_mean_ms": round(float(np.mean(all_tpot)), 2) if all_tpot else None,
        "tpot_p95_ms": round(np.percentile(all_tpot, 95), 2) if all_tpot else None,
        "tpot_p99_ms": round(np.percentile(all_tpot, 99), 2) if all_tpot else None,
        # None (not 0) when there's no data -- 0 would read as "zero latency between tokens".
        "itl_p95_ms": round(np.percentile(all_itls, 95), 2) if all_itls else None,
        "itl_p99_ms": round(np.percentile(all_itls, 99), 2) if all_itls else None,
        "latency_e2e_p50_ms": round(np.percentile(all_e2e, 50), 2),
        "latency_e2e_p95_ms": round(np.percentile(all_e2e, 95), 2),
        "latency_e2e_p99_ms": round(np.percentile(all_e2e, 99), 2),
        "vram_base_mb": base_vram,
        "vram_peak_mb": peak_vram,
        "vram_delta_mb": vram_delta,
        "vram_avg_mb": avg_vram,
        "vram_samples_collected": len(vram_samples),
        "gpu_util_avg": average_gpu_util,
        "gpu_util_max": maximum_gpu_util,
        "errors": dict(error_counter),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="JSONL file, one {\"text\": ...} object per line")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5, help="Untimed requests run before measurement starts")
    parser.add_argument("--max-tokens", type=int, default=256, help="Generation cap per request")
    parser.add_argument("--ignore-eos", action="store_true",
                         help="Force generation to run the full --max-tokens length (vLLM/SGLang), "
                              "for clean TPS comparisons unconfounded by early stopping")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--tag", type=str, default="", help="Run identifier, e.g. vllm_fp8_kvcache")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--system-prompt-file", type=str, default=None,
                     help="Fixed text sent as a system-role message on every request")
    parser.add_argument("--response-format-file", type=str, default=None,
                     help="JSON file containing an OpenAI-style response_format object")
    args = parser.parse_args()
    system_prompt = open(args.system_prompt_file).read().strip() if args.system_prompt_file else None
    response_format = json.load(open(args.response_format_file)) if args.response_format_file else None
    final_metrics = asyncio.run(
        benchmark(args.endpoint, args.model, args.dataset, args.concurrency,
                  args.requests, args.warmup, args.tag,
                  args.max_tokens, args.ignore_eos, args.gpu_index,
                  system_prompt=system_prompt, response_format=response_format)
    )

    with open(args.output, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(json.dumps(final_metrics, indent=2))
import asyncio
import aiohttp
import time
import json
import numpy as np
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RequestMetrics:
    success: bool
    ttft: Optional[float] = None  # Time to first token
    total_time: Optional[float] = None
    output_tokens: int = 0
    error: Optional[str] = None

class AsyncLoadTester:
    def __init__(self, endpoint: str, model_name: str, concurrency: int, api_key: str = None, timeout: int = 60):
        self.endpoint = endpoint
        self.model_name = model_name
        self.concurrency = concurrency
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.results: List[RequestMetrics] = []

    async def _send_request(self, session: aiohttp.ClientSession, prompt: str, max_tokens: int) -> RequestMetrics:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_time = time.perf_counter()
        first_token_time = None
        output_tokens = 0
        
        async with self.semaphore:
            try:
                async with session.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        return RequestMetrics(success=False, error=f"HTTP {response.status}: {error_body}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or line == "data: [DONE]":
                            continue
                        
                        if line.startswith("data: "):
                            if first_token_time is None:
                                first_token_time = time.perf_counter() - start_time
                            output_tokens += 1

                total_time = time.perf_counter() - start_time
                return RequestMetrics(
                    success=True,
                    ttft=first_token_time,
                    total_time=total_time,
                    output_tokens=output_tokens
                )

            except Exception as e:
                return RequestMetrics(success=False, error=str(e))

    async def run_benchmark(self, prompts: List[str], max_tokens: int = 256):
        print(f"Starting benchmark: {len(prompts)} requests with concurrency {self.concurrency}...")
        
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._send_request(session, prompt, max_tokens)
                for prompt in prompts
            ]
            self.results = await asyncio.gather(*tasks)

    def generate_report(self) -> dict:
        successful_reqs = [r for r in self.results if r.success]
        errors = [r for r in self.results if not r.success]

        if not successful_reqs:
            print("All requests failed!")
            if errors:
                print(f"Sample error: {errors[0].error}")
            return {"error": "All requests failed"}

        ttfts = [r.ttft for r in successful_reqs if r.ttft is not None]
        total_times = [r.total_time for r in successful_reqs if r.total_time is not None]
        total_tokens = sum(r.output_tokens for r in successful_reqs)
        total_benchmark_time = max(total_times) if total_times else 0.01

        tps = total_tokens / total_benchmark_time
        req_rate = len(successful_reqs) / total_benchmark_time

        print("\n--- Benchmark Results ---")
        print(f"Total Requests:      {len(self.results)}")
        print(f"Successful Requests: {len(successful_reqs)}")
        print(f"Failed Requests:     {len(errors)}")
        print(f"Total Tokens Gen:    {total_tokens}")
        print(f"Throughput (TPS):    {tps:.2f} tokens/s")
        print(f"Request Rate:        {req_rate:.2f} req/s")
        
        if ttfts:
            print("\n--- Time to First Token (TTFT) ---")
            print(f"Mean: {np.mean(ttfts):.4f}s")
            print(f"P50:  {np.percentile(ttfts, 50):.4f}s")
            print(f"P95:  {np.percentile(ttfts, 95):.4f}s")

        if total_times:
            print("\n--- End-to-End Latency ---")
            print(f"Mean: {np.mean(total_times):.4f}s")
            print(f"P50:  {np.percentile(total_times, 50):.4f}s")
            print(f"P95:  {np.percentile(total_times, 95):.4f}s")

        # Compile metrics for JSON output
        return {
            "total_requests": len(self.results),
            "successful_requests": len(successful_reqs),
            "failed_requests": len(errors),
            "throughput_tps": round(tps, 2),
            "ttft_p50": round(np.percentile(ttfts, 50), 4) if ttfts else None,
            "ttft_p95": round(np.percentile(ttfts, 95), 4) if ttfts else None,
            "latency_p50": round(np.percentile(total_times, 50), 4) if total_times else None,
            "latency_p95": round(np.percentile(total_times, 95), 4) if total_times else None,
        }

def main():
    parser = argparse.ArgumentParser(description="LLM Async Load Tester")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/chat/completions", help="API endpoint")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max output tokens per request")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON file containing a list of prompts")
    parser.add_argument("--requests", type=int, default=50, help="Total number of requests (used if no dataset provided)")
    parser.add_argument("--output", type=str, default=None, help="Path to save benchmark results as JSON")
    
    args = parser.parse_args()

    # Load dataset if provided, else fallback to dummies
    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset, 'r') as f:
            prompts = json.load(f)
            # Truncate or expand to match --requests limit if desired, or just use all
            prompts = prompts[:args.requests] if len(prompts) > args.requests else prompts
    else:
        print("Warning: No dataset provided. Falling back to dummy prompts.")
        prompts = [f"Explain the theory of relativity. Request ID: {i}" for i in range(args.requests)]

    tester = AsyncLoadTester(
        endpoint=args.endpoint,
        model_name=args.model,
        concurrency=args.concurrency
    )
    
    asyncio.run(tester.run_benchmark(prompts, args.max_tokens))
    metrics_summary = tester.generate_report()

    # Save logic for the summary dictionary
    if args.output and "error" not in metrics_summary:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nResults successfully saved to {args.output}")

if __name__ == "__main__":
    main()
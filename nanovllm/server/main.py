from __future__ import annotations

import argparse

import uvicorn

from .config import ServerConfig
from .gateway import create_app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Nano-vLLM with an OpenAI-compatible API")
    parser.add_argument("--model", required=True, help="Path to the HuggingFace model directory")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel degree")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override maximum model length")
    parser.add_argument("--max-concurrent-requests", type=int, default=32, help="Maximum in-flight requests")
    parser.add_argument("--max-queue", type=int, default=128, help="Maximum queued requests before rejecting")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Request timeout hint (seconds)")
    parser.add_argument("--disable-streaming", action="store_true", help="Disable streaming responses")
    parser.add_argument("--api-key", default=None, help="Static API key required for requests")
    parser.add_argument("--metrics-endpoint", default=None, help="Enable Prometheus metrics at the given path")
    parser.add_argument("--health-endpoint", default="/healthz", help="Health probe endpoint path")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Uvicorn log level"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ServerConfig(
        model=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_concurrent_requests=args.max_concurrent_requests,
        max_queue_size=args.max_queue,
        request_timeout_s=args.request_timeout,
        enable_streaming=not args.disable_streaming,
        api_key=args.api_key,
        log_level=args.log_level,
        metrics_endpoint=args.metrics_endpoint,
        health_endpoint=args.health_endpoint,
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()

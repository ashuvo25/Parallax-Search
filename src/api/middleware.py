"""Request logging middleware: request ID, timing, one JSON log line per request,
and a safe 500 response for any unexpected error."""
import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.errors import error_body

req_log = logging.getLogger("api.requests")


async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    request.state.log_extra = {}
    start = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        # a bug in our code: log everything, show the user nothing internal
        status = 500
        req_log.exception("unhandled error",
                          extra={"fields": {"request_id": request_id, "path": request.url.path}})
        response = JSONResponse(status_code=500, content=error_body(
            request, 500, "Internal server error. Quote the request_id when reporting it."))

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id

    fields = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "latency_ms": latency_ms,
        "client": request.client.host if request.client else None,
        **request.state.log_extra,
    }
    level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
    req_log.log(level, f"{request.method} {request.url.path} {status}", extra={"fields": fields})
    return response
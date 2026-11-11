"""One error format for every error response, plus the handlers that produce it."""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.llm import LLMError

ERROR_TYPES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "bad_gateway",
    503: "service_unavailable",
    504: "gateway_timeout",
}


def error_body(request: Request, status: int, message: str, details: list | None = None) -> dict:
    error = {
        "code": status,
        "type": ERROR_TYPES.get(status, "error"),
        "message": message,
        "request_id": getattr(request.state, "request_id", None),
    }
    if details:
        error["details"] = details
    return {"error": error}


def _note(request: Request, **fields) -> None:
    """Add fields to this request's log line."""
    if not hasattr(request.state, "log_extra"):
        request.state.log_extra = {}
    request.state.log_extra.update(fields)


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    _note(request, error=str(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(request, exc.status_code, str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()

    # broken JSON is a bad request (400), not a field problem (422)
    if any(e.get("type") == "json_invalid" for e in errors):
        _note(request, error="invalid JSON")
        return JSONResponse(status_code=400,
                            content=error_body(request, 400, "Request body is not valid JSON."))

    details = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
         "message": e["msg"]}
        for e in errors
    ]
    _note(request, error="validation", fields=[d["field"] for d in details])
    return JSONResponse(status_code=422,
                        content=error_body(request, 422, "Request validation failed.", details))


async def llm_error_handler(request: Request, exc: LLMError):
    _note(request, error=type(exc).__name__)
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return JSONResponse(status_code=exc.status,
                        content=error_body(request, exc.status, str(exc)),
                        headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(LLMError, llm_error_handler)
import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import sys
import uuid
from typing import Any, Dict, Optional

import yaml

from elliot.utils.folder import check_dir, check_path, path_absolute, path_joiner

DEFAULT_SETTINGS: Dict[str, Any] = {
    "app_name": "elliot",
    "console_level": "INFO",
    "file_level": "DEBUG",
    "max_bytes": 5_000_000,
    "backup_count": 5,
    "json_file_logs": True,
    "include_location": True,
    "color_console": True,
}

LOG_RECORD_BASE_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "stacklevel",
}


class _LoggerState:
    def __init__(self) -> None:
        self.run_id: str = ""
        self.log_dir: str = ""
        self.log_file: str = ""
        self.settings: Dict[str, Any] = {}
        self.initialized: bool = False


STATE = _LoggerState()


def _generate_run_id() -> str:
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{timestamp}-{suffix}"


def _coerce_level(value: Any, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return logging._nameToLevel.get(value.upper(), fallback)
    return fallback


def _load_settings(path_config: Optional[str]) -> Dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if path_config and check_path(path_config):
        with open(path_config, "r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
            if isinstance(loaded, dict):
                settings.update(loaded)
    return settings


class StructuredFormatter(logging.Formatter):
    LEVEL_EMOJI = {
        logging.DEBUG: "🧩",
        logging.INFO: "🟢",
        logging.WARNING: "⚠️",
        logging.ERROR: "💥",
        logging.CRITICAL: "🛑",
    }

    def __init__(
        self,
        *,
        json_output: bool,
        app_name: str,
        run_id: str,
        include_location: bool = True,
        color: bool = True,
    ) -> None:
        super().__init__()
        self.json_output = json_output
        self.app_name = app_name
        self.run_id = run_id
        self.include_location = include_location
        self.color = color and not json_output

    @staticmethod
    def _utc_timestamp(record: logging.LogRecord) -> str:
        return (
            datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(timespec="milliseconds")
        )

    def _extract_extras(self, record: logging.LogRecord) -> Dict[str, Any]:
        extras = {}
        for key, value in record.__dict__.items():
            if key in LOG_RECORD_BASE_KEYS:
                continue
            if key in {"component", "run_id", "context"}:
                continue
            extras[key] = value
        return extras

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self._utc_timestamp(record),
            "run_id": getattr(record, "run_id", self.run_id),
            "app": self.app_name,
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "message": record.getMessage(),
        }
        emoji = self.LEVEL_EMOJI.get(record.levelno, "🧭")

        if self.include_location:
            payload["where"] = f"{record.filename}:{record.lineno}"

        context = getattr(record, "context", None)
        if context:
            payload["context"] = context

        extras = self._extract_extras(record)
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if self.json_output:
            return json.dumps(payload, default=str)

        ts_display = payload["ts"]
        level_display = f"{emoji} {record.levelname:<8}"
        component_display = payload["component"]
        message_text = payload["message"]
        if self.color:
            ts_display = self._colorize(ts_display, record.levelno)
            level_display = self._colorize(level_display, record.levelno)
            component_display = self._colorize(component_display, record.levelno)
            message_text = self._colorize(message_text, record.levelno)

        human_parts = [
            ts_display,
            level_display,
            component_display,
            message_text,
        ]

        if context:
            human_parts.append(f"context={json.dumps(context, default=str)}")
        if extras:
            human_parts.append(f"extra={json.dumps(extras, default=str)}")
        if record.exc_info:
            human_parts.append("exception=see file log")

        return " | ".join(human_parts)

    def _colorize(self, text: str, level: int) -> str:
        colors = {
            logging.DEBUG: "\033[36m",
            logging.INFO: "\033[32m",
            logging.WARNING: "\033[33m",
            logging.ERROR: "\033[31m",
            logging.CRITICAL: "\033[41m\033[97m",
        }
        reset = "\033[0m"
        return f"{colors.get(level, '')}{text}{reset if colors.get(level) else ''}"


class StructuredLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> Any:
        extra = kwargs.pop("extra", {}) or {}
        context_extra = extra.pop("context", {})
        base_context = self.extra.get("context", {})

        context = {}
        if isinstance(base_context, dict):
            context.update(base_context)
        if isinstance(context_extra, dict):
            context.update(context_extra)

        merged_extra = {k: v for k, v in self.extra.items() if k != "context"}
        merged_extra.update(extra)

        if context:
            merged_extra["context"] = context

        merged_extra.setdefault("run_id", self.extra.get("run_id") or STATE.run_id)
        merged_extra.setdefault("component", self.logger.name)

        kwargs["extra"] = merged_extra
        return msg, kwargs


def _ensure_state() -> None:
    if not STATE.run_id:
        STATE.run_id = _generate_run_id()
    if not STATE.settings:
        STATE.settings = dict(DEFAULT_SETTINGS)


def _configure_root_handlers(settings: Dict[str, Any]) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    app_name = settings.get("app_name", "elliot")
    include_location = bool(settings.get("include_location", True))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings["console_level"])
    console_handler.setFormatter(
        StructuredFormatter(
            json_output=False,
            app_name=app_name,
            run_id=STATE.run_id,
            include_location=include_location,
            color=bool(settings.get("color_console", True)),
        )
    )

    file_handler = RotatingFileHandler(
        STATE.log_file,
        maxBytes=settings["max_bytes"],
        backupCount=settings["backup_count"],
        encoding="utf-8",
    )
    file_handler.setLevel(settings["file_level"])
    file_handler.setFormatter(
        StructuredFormatter(
            json_output=settings["json_file_logs"],
            app_name=app_name,
            run_id=STATE.run_id,
            include_location=include_location,
        )
    )

    root.addHandler(console_handler)
    root.addHandler(file_handler)


def init(path_config: str, folder_log: str, log_level: int = logging.INFO) -> None:
    settings = _load_settings(path_config)
    settings["console_level"] = _coerce_level(
        settings.get("console_level"), log_level
    )
    settings["file_level"] = _coerce_level(settings.get("file_level"), log_level)
    settings["max_bytes"] = int(settings.get("max_bytes", DEFAULT_SETTINGS["max_bytes"]))
    settings["backup_count"] = int(
        settings.get("backup_count", DEFAULT_SETTINGS["backup_count"])
    )
    settings["json_file_logs"] = bool(
        settings.get("json_file_logs", DEFAULT_SETTINGS["json_file_logs"])
    )

    STATE.settings = settings
    STATE.log_dir = path_absolute(folder_log)
    check_dir(STATE.log_dir)

    if not STATE.run_id:
        STATE.run_id = settings.get("run_id") or _generate_run_id()

    STATE.log_file = path_joiner(
        STATE.log_dir, f"{settings.get('app_name', 'elliot')}-{STATE.run_id}.log"
    )

    _configure_root_handlers(settings)
    STATE.initialized = True


def get_logger(name: str, log_level: int = logging.DEBUG, **context: Any) -> logging.LoggerAdapter:
    _ensure_state()
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return StructuredLoggerAdapter(logger, {"context": context, "run_id": STATE.run_id})


def get_logger_model(name: str, log_level: int = logging.DEBUG) -> logging.LoggerAdapter:
    return get_logger(name, log_level=log_level, model=name)


def prepare_logger(
    name: str, path: str, log_level: int = logging.DEBUG
) -> logging.LoggerAdapter:
    _ensure_state()
    check_dir(path)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    for handler in list(logger.handlers):
        if getattr(handler, "_elliot_model_handler", False):
            logger.removeHandler(handler)

    logfilepath = path_absolute(
        path_joiner(path, f"{name}-{STATE.run_id}.log")
    )

    handler = RotatingFileHandler(
        logfilepath,
        maxBytes=STATE.settings.get("max_bytes", DEFAULT_SETTINGS["max_bytes"]),
        backupCount=STATE.settings.get("backup_count", DEFAULT_SETTINGS["backup_count"]),
        encoding="utf-8",
    )
    handler.setLevel(log_level)
    handler.setFormatter(
        StructuredFormatter(
            json_output=True,
            app_name=STATE.settings.get("app_name", "elliot"),
            run_id=STATE.run_id,
            include_location=STATE.settings.get("include_location", True),
            color=False,
        )
    )
    handler._elliot_model_handler = True  # type: ignore[attr-defined]

    logger.addHandler(handler)
    logger.propagate = True

    return StructuredLoggerAdapter(
        logger, {"context": {"model": name}, "run_id": STATE.run_id}
    )

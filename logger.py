#!/usr/bin/env python3
"""
Logging configuration for AutonomousBoat

This module sets up Python's logging with both console and rotating-file handlers,
using parameters from config.yaml. File logs are emitted in JSON format for easy parsing.
"""
import os
import yaml
import logging
import json
from logging.handlers import RotatingFileHandler

# Load logging configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        _full_cfg = yaml.safe_load(f)
    _log_cfg = _full_cfg.get('logging', {}) or {}
except Exception:
    _log_cfg = {}

# Ensure output folder exists
_output_folder = _log_cfg.get('output_folder', 'logs/')
os.makedirs(_output_folder, exist_ok=True)

# Determine log level
_level = getattr(logging, _log_cfg.get('level', 'INFO').upper(), logging.INFO)

# Configure root logger
_logger = logging.getLogger()
_logger.setLevel(_level)

# Console handler
_ch = logging.StreamHandler()
_ch.setLevel(_level)
_console_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
_ch.setFormatter(_console_fmt)
_logger.addHandler(_ch)

# Rotating file handler with JSON output
_log_file = os.path.join(_output_folder, 'autonomous_boat.log')
_fh = RotatingFileHandler(_log_file,
                          maxBytes=_log_cfg.get('max_bytes', 5_000_000),
                          backupCount=_log_cfg.get('backup_count', 3))
_fh.setLevel(_level)

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
        }
        # include any extra fields if provided
        if hasattr(record, 'extra_data'):
            data.update(record.extra_data)
        return json.dumps(data)

_json_fmt = _JsonFormatter()
_fh.setFormatter(_json_fmt)
_logger.addHandler(_fh)

# Expose the configured logger
logger = _logger

# Convenience function for structured logging
def log_structured(level: int, msg: str, **extra_data) -> None:
    """Log a message with additional structured data."""
    _logger.log(level, msg, extra={'extra_data': extra_data})

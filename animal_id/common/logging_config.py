"""Single place the project configures logging.

Entry points (scripts, CLIs) call :func:`setup_logging` once and use the logger it
returns. Library modules never configure logging themselves — they just do
``logger = logging.getLogger(__name__)`` and log, so output stays under the
caller's control.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(name: str, stdout_level: int | None = None) -> logging.Logger:
    """Log to stdout at ``stdout_level`` (None leaves stdout silent); returns ``name``'s logger."""
    if stdout_level is not None:
        logging.basicConfig(
            level=stdout_level,
            format=LOG_FORMAT,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    return logging.getLogger(name)

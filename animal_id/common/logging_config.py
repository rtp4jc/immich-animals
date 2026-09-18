"""Single place the project configures logging.

Entry points (scripts, CLIs) call :func:`setup_logging` once. Library modules never
configure logging themselves — they just do ``logger = logging.getLogger(__name__)``
and log, so output stays under the caller's control.
"""

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for a command-line entry point."""
    logging.basicConfig(
        level=level, format=LOG_FORMAT, handlers=[logging.StreamHandler()]
    )

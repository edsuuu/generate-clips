import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    return logging.getLogger("auto-post")


logger = setup_logging()

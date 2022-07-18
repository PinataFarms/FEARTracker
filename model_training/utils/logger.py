import logging
import os

import coloredlogs


COLOREDLOGS_FIELD_STYLES = coloredlogs.DEFAULT_FIELD_STYLES
COLOREDLOGS_FIELD_STYLES.update(
    {
        "asctime": {"color": "green"},
        "filename": {"color": "green"},
        "fileno": {"color": "green"},
    }
)


def create_logger(
    name: str,
    msg_format: str = "",
) -> logging.Logger:
    msg_format = msg_format or "%(asctime)s %(hostname)s %(name)s %(levelname)s - %(message)s - %(filename)s:%(lineno)d"
    logger = logging.Logger(name)
    console_handler = logging.StreamHandler()
    level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    coloredlogs.install(
        level=level,
        logger=logger,
        field_styles=COLOREDLOGS_FIELD_STYLES,
        fmt=msg_format,
    )

    return logger


logger = create_logger(__name__)

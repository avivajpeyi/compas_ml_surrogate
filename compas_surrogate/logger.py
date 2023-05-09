import sys
import warnings

from loguru import logger

logger.remove(0)
logger.add(
    sys.stderr,
    format="|<blue>COMPAS-SUR</>|{time:DD/MM HH:mm:ss}|{level}| {message} ",
    colorize=True,
    level="INFO",
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

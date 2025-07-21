#set_up_loggers.py
import logging
import os
import sys
from pathlib import Path

def setup_logger(logger: logging.Logger, config: dict = {"file": "INFO",
                                                       "console": "ERROR"}):
    
    if not os.path.exists(Path(os.getcwd()) / "logs"):
        os.makedirs(Path(os.getcwd()) / "logs")
    
    level_console = config["console"]
    level_file = config["file"]
    try:
        logger.setLevel(level_console)
    except ValueError as e:
        raise ValueError(f"{e} - change config value")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(Path("logs") / f"{logger.name}.log")
    file_handler.setLevel(level_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_console) # Log everything to the console for development.
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
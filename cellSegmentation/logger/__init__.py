import logging
import os
import sys
from datetime import datetime
from from_root import from_root

# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE = "running_logs.log"

log_path = os.path.join(from_root(), 'log', LOG_FILE)

os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging_str = "[%(asctime)s ] %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
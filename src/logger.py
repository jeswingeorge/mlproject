import logging, os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) 
os.makedirs(logs_path, exist_ok=True)

## create path for log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

## create logger using basicConfig
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# if __name__ == "__main__":
#     logging.info("Logger initialized successfully.")


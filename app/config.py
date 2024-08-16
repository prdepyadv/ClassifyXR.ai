from os import path
import os
from dotenv import load_dotenv
import urllib


def str_to_bool(value, default=False):
    return str(value).lower() in ["true", "1", "t", "y", "yes"]


dotenv_path = path.join(path.dirname(path.dirname(path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

SCHEDULER_EXECUTORS = {"default": {"type": "threadpool", "max_workers": 20}}
SCHEDULER_JOB_DEFAULTS = {"coalesce": False}

FLASK_RUN_HOST = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
FLASK_RUN_PORT = int(os.getenv("FLASK_RUN_PORT", 8080))

FLASK_APP = os.getenv("FLASK_APP", "app/run")
FLASK_ENV = os.getenv("FLASK_ENV", "production")
FLASK_DEBUG = str_to_bool(os.getenv("FLASK_DEBUG", "False"))
DEBUG = str_to_bool(os.getenv("FLASK_DEBUG", "False"))

USE_SSL = str_to_bool(os.getenv("USE_SSL", "False"))

SECRET_KEY = os.getenv("SECRET_KEY")

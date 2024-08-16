from abc import ABC, ABCMeta, abstractmethod
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_apscheduler import APScheduler
from app.classification_system import ClassificationSystem
from app.commands import define_tasks
import logging
import threading


class SingletonMeta(ABCMeta):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseLogger(ABC, metaclass=SingletonMeta):
    @abstractmethod
    def debug(self, message):
        pass

    @abstractmethod
    def info(self, message):
        pass

    @abstractmethod
    def warning(self, message):
        pass

    @abstractmethod
    def error(self, message):
        pass

    @abstractmethod
    def critical(self, message):
        pass


class Logger(BaseLogger):
    _logger = None

    def __init__(self):
        self._initiate_logger()

    def _initiate_logger(self):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        file_handler = RotatingFileHandler(
            "error.log", maxBytes=10000000, backupCount=5
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    def debug(self, message):
        if self._logger is not None:
            self._logger.debug(message)

    def info(self, message):
        if self._logger is not None:
            self._logger.info(message)

    def warning(self, message):
        if self._logger is not None:
            self._logger.warning(message)

    def error(self, message):
        if self._logger is not None:
            self._logger.error(message)

    def critical(self, message):
        if self._logger is not None:
            self._logger.critical(message)


def create_app():
    app = Flask(__name__)
    app.config.from_pyfile("config.py")

    logger = Logger()
    logger.info("Application created.")

    ticket1 = """
    I ordered a laptop from your store last week (Order #12345), but I received a tablet instead. 
    This is unacceptable! I need the laptop for work urgently. Please resolve this immediately or I'll have to dispute the charge.
    """

    ticket2 = """
    Hello, I'm having trouble logging into my account. I've tried resetting my password, but I'm not receiving the reset email. 
    Can you please help me regain access to my account? I've been a loyal customer for years and have several pending orders.
    """

    classification_system = ClassificationSystem()
    print(classification_system.classify_ticket(ticket1))
    print(classification_system.classify_ticket(ticket2))

    # scheduler = APScheduler()
    # scheduler.init_app(app)
    # define_tasks(scheduler=scheduler, app=app)
    # scheduler.start()

    return app

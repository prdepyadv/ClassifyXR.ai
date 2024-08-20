from abc import ABC, ABCMeta, abstractmethod
import json
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify, request
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

    classification_system = ClassificationSystem()

    @app.route('/classify/ticket', methods = ['POST']) 
    def classify_ticket():
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": True, "message": "Invalid request, 'message' field is missing"}), 400

        example_ticket = """
        I need to cancel my subscription, but the website doesn't allow me to do so. I've tried multiple times,
        but I keep getting error messages. Can someone assist me with this cancellation?
        """

        example_ticket = """
        Our team has noticed that the website has been extremely slow recently.
        The page load time has increased from 1.5 seconds to almost 6 seconds, which is unacceptable.
        This slowdown has led to a 25% drop in our daily transactions. Can you please look into this urgently?
        """

        ticket = data['message']
        try:
            classification = classification_system.classify(ticket)
        except Exception as e:
            logger.error(f"Error classifying ticket: {e}")
            return jsonify({"error": True, "message": "Error classifying ticket"}), 500

        return jsonify({ "error": False, "data": classification})

    return app

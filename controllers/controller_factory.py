# controllers/controller_factory.py
from .base_controller import BaseController
from .simple_controller import SimpleController
from .pid_controller import PIDController
from config import config


def create_controller() -> BaseController:
    controller_type = config.get("Controller", "controller_type", "simple").lower()

    if controller_type == "pid":
        return PIDController()
    elif controller_type == "simple":
        return SimpleController()
    else:
        print(f"[Controller] 未知类型 {controller_type}，回退使用 simple")
        return SimpleController()
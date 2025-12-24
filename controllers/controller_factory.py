# controllers/controller_factory.py
from typing import Optional
from .base_controller import BaseController
from .simple_controller import SimpleController
from .pid_controller import PIDController
from config import config


class ControllerFactory:
    """
    控制器创建工厂
    目前支持 simple 和 pid 两种控制器
    未来可扩展 advanced / custom 等类型
    """

    @staticmethod
    def create() -> BaseController:
        controller_type = config.getstr("Controller", "controller_type", "simple").lower()

        controllers_map = {
            "simple": SimpleController,
            "pid": PIDController,
        }

        controller_class = controllers_map.get(controller_type)

        if controller_class is None:
            print(f"[ControllerFactory] 未知控制器类型: {controller_type}，回退使用 simple")
            return SimpleController()

        return controller_class()


def get_controller() -> BaseController:
    """
    全局获取控制器的便捷接口
    可在 main.py 或其他地方直接调用
    """
    return ControllerFactory.create()
# controllers/controller_factory.py - 控制器工厂
#
# 核心职责：
# 1. 根据配置创建相应的控制器实例
# 2. 管理控制器类型映射
# 3. 提供全局控制器获取接口
#
# 架构特点：工厂模式，支持多种控制器类型
#

from typing import Optional
from .base_controller import BaseController
from .simple_controller import SimpleController
from .pid_controller import PIDController
from .pro_controller import PROController
from config import config
from utils.logger import get_logger

_log = get_logger("ControllerFactory")


class ControllerFactory:
    """
    控制器创建工厂
    目前支持 simple 和 pid 两种控制器
    未来可扩展 advanced / custom 等类型
    """

    @staticmethod
    def create() -> BaseController:
        """根据配置创建控制器实例"""
        # 从配置中获取控制器类型
        controller_type = config.getstr("Controller", "controller_type", "simple").lower()

        # 控制器类型映射表
        controllers_map = {
            "simple": SimpleController,  # 简单比例控制器
            "pid": PIDController,   # PID控制器
            "pro": PROController,   # PRO控制器
        }

        # 获取对应控制器类
        controller_class = controllers_map.get(controller_type)

        if not controller_class:
            _log.warning("未知控制器类型: %s，回退使用 simple", controller_type)
            return SimpleController()

        _log.info("加载控制器: %sController", controller_type.upper())
        return controller_class()  # 直接实例化

def get_controller() -> BaseController:
    """
    全局获取控制器的便捷接口
    可在 main.py 或其他地方直接调用
    """
    return ControllerFactory.create()
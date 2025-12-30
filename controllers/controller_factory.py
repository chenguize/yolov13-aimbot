# controllers/controller_factory.py - 控制器工厂
from typing import Optional, Tuple, Any
from config import config

# 引入控制器实现
from .base_controller import BaseController
from .simple_controller import SimpleController
from .pro_controller import PROController

# 如果有 pid_controller 也引用进来，没有就算了
try:
    from .pid_controller import PIDController
except ImportError:
    PIDController = None

# 引入拟人化模块
from .humanize.humanize import Humanizer


class HumanizedControllerWrapper(BaseController):
    """
    [Decorator] 拟人化装饰器
    包装任意基础控制器，在 compute 输出后应用 Humanizer 滤镜。
    """

    def __init__(self, base_controller: BaseController):
        # 继承 BaseController 主要是为了类型兼容，实际逻辑全代理给 base
        # 注意：这里不调用 super().__init__() 避免重复读取 config
        self.base = base_controller
        self.humanizer = Humanizer()

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        # 1. 调用核心控制器计算 (如 PID)
        raw_x, raw_y = self.base.compute(intent_dx, intent_dy, dt)

        # 2. 应用拟人化滤镜 (加噪/抖动)
        final_x, final_y = self.humanizer.apply(raw_x, raw_y)

        return final_x, final_y

    def should_trigger(self, target, screen_center) -> bool:
        # 直接透传给基础控制器处理
        return self.base.should_trigger(target, screen_center)


class ControllerFactory:
    """
    控制器创建工厂
    """

    @staticmethod
    def create() -> BaseController:
        """根据配置创建控制器实例"""
        controller_type = config.getstr("Controller", "controller_type", "simple").lower()

        # 1. 实例化核心控制器
        controller: Optional[BaseController] = None

        if controller_type == "pro":
            controller = PROController()
        elif controller_type == "pid" and PIDController:
            controller = PIDController()
        else:
            if controller_type != "simple":
                print(f"[Factory] 未知控制器类型: {controller_type}，回退使用 simple")
            controller = SimpleController()

        # 2. 检查是否开启 Humanize
        # 如果开启，就给控制器套上一层壳
        if config.getbool("Humanize", "enable_humanize", False):
            print(f"[Factory] 🧬 已启用拟人化后处理 (Humanized + {controller_type})")
            return HumanizedControllerWrapper(controller)
        else:
            print(f"[Factory] 使用标准控制器: {controller_type}")
            return controller


def get_controller() -> BaseController:
    return ControllerFactory.create()
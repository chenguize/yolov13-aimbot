# aim_strategies/factory.py
from config import config
from utils.logger import get_logger

_log = get_logger("AimStrategyFactory")


def create_aim_strategy():
    """
    根据配置创建游戏专用瞄准策略实例
    """
    game = config.getstr("General", "current_game", "valorant").lower()
    package = config.getstr("AimStrategy", "strategy_package", game)

    try:
        if package == "valorant":
            from aim_strategies.valorant.strategy import ValorantStrategy
            return ValorantStrategy()
        else:
            _log.warning("未找到策略包 '%s'，回退使用 valorant", package)
            from aim_strategies.valorant.strategy import ValorantStrategy
            return ValorantStrategy()
    except ImportError as e:
        _log.error("策略加载失败: %s", e)
        return None
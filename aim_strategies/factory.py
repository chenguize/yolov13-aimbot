# aim_strategies/factory.py
from typing import Optional
from config import config


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
            print(f"[StrategyFactory] 未找到策略包 '{package}'，回退使用 valorant")
            from aim_strategies.valorant.strategy import ValorantStrategy
            return ValorantStrategy()
    except ImportError as e:
        print(f"[StrategyFactory] 加载失败: {e}")
        return None
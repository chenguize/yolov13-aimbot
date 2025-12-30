import pickle
import time
from pathlib import Path
import numpy as np

# 自动定位 debug_trace.pkl
pkl_path = Path(__file__).resolve().parent.parent / "debug_trace.pkl"


def safe_get(obj, attr, default="N/A"):
    """安全获取属性，防止报错"""
    val = getattr(obj, attr, default)
    if isinstance(val, (tuple, list, np.ndarray)) and len(val) == 2:
        return f"({val[0]:.1f}, {val[1]:.1f})"
    if isinstance(val, float):
        return f"{val:.2f}"
    return val


def analyze():
    if not pkl_path.exists():
        print(f"❌ 未找到日志文件: {pkl_path}")
        return

    print(f"📂 正在分析: {pkl_path} ...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # 过滤出“锁敌状态”的帧
    valid_frames = [ctx for ctx in data if getattr(ctx, 'is_valid', False)]
    total = len(data)
    valid = len(valid_frames)

    print(f"📊 数据概览: 总帧数 {total} | 锁敌帧数 {valid}")
    print("-" * 100)
    print(
        f"{'Frame':<6} | {'Time':<8} | {'Target (预测点)':<18} | {'Vel (速度)':<12} | {'Mouse (指令)':<16} | {'Conf':<5}")
    print("-" * 100)

    if valid == 0:
        print("⚠️ 没有有效的锁敌帧 (is_valid=True)，无法分析漂移原因。")
        return

    # 打印前 30 帧有效数据，以及最后 5 帧
    targets_to_print = valid_frames[:30]

    start_time = getattr(targets_to_print[0], 't_cap', 0)

    for i, ctx in enumerate(targets_to_print):
        # 1. 基础信息
        fid = getattr(ctx, 'frame_id', i)  # 如果没有 frame_id 属性就用索引
        t_cap = getattr(ctx, 't_cap', 0)
        rel_time = (t_cap - start_time) * 1000 if start_time else 0

        # 2. 预测点 (WorldModel 输出) -> 决定了"想去哪"
        # 格式通常是 (x, y) 0~256
        pred = safe_get(ctx, 'p_predict', (0, 0))

        # 3. 速度 (Kalman)
        vel = safe_get(ctx, 'v_real', (0, 0))

        # 4. 鼠标指令 (Strategy 输出) -> 决定了"怎么动"
        # main.py 里通常把结果存为 ctx.strategy_result
        # 可能是 tuple (dx, dy) 或 对象
        strat_res = getattr(ctx, 'strategy_result', None)
        mouse_str = "None"
        if strat_res:
            if isinstance(strat_res, (tuple, list)):
                mouse_str = f"({strat_res[0]:.0f}, {strat_res[1]:.0f})"
            elif hasattr(strat_res, 'raw_counts_x'):  # ValorantStrategy 对象
                mouse_str = f"({strat_res.raw_counts_x:.0f}, {strat_res.raw_counts_y:.0f})"
            else:
                mouse_str = str(strat_res)

        # 5. 置信度
        conf = getattr(ctx, 'conf', 0.0)

        print(f"{fid:<6} | +{rel_time:<6.0f}ms | {str(pred):<18} | {str(vel):<12} | {mouse_str:<16} | {conf:.2f}")

    print("-" * 100)
    print("💡 分析指南:")
    print("1. 看 [Target]: 截图中心是 128。如果 y < 128 (如 30)，目标在上面；y > 128 (如 170)，目标在下面。")
    print("2. 看 [Mouse]: 正数是向右/下移动，负数是向左/上移动。")
    print("   👉 如果 Target y=170 (下)，但 Mouse y 是负数 (上)，那就是轴反了 -> 鼠标就会飞天！")
    print("-" * 100)


if __name__ == "__main__":
    analyze()
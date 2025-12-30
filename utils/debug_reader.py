import pickle
from pathlib import Path

# 定位上一级目录的文件
pkl_path = Path(__file__).resolve().parent.parent / "debug_trace.pkl"


def filter_trace():
    if not pkl_path.exists():
        print("未找到 pkl 文件")
        return

    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)

    # 核心过滤逻辑：仅保留 is_valid 为 True 的帧
    # 如果你的 data 是 InferenceContext 对象列表
    filtered_data = [frame for frame in raw_data if getattr(frame, 'is_valid', False)]

    print(f"📊 原始总帧数: {len(raw_data)}")
    print(f"🎯 有效锁敌帧数: {len(filtered_data)}")

    # 打印前 5 个有效坐标观察参数
    for i, frame in enumerate(filtered_data[:5]):
        print(f"帧 {i}: ID={frame.selected_id}, 预测点={frame.p_predict}, 速度={frame.v_real}")


if __name__ == "__main__":
    filter_trace()
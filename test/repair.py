from ultralytics import YOLO
import os

def export_to_onnx(pt_path, onnx_path, imgsz=256):
    """导出 YOLO pt → ONNX"""
    model = YOLO(pt_path)

    # 直接禁用 simplify！
    # onnxslim 在处理某些特殊算子时会导致 C++ 级别的内存越界(0xC0000005)，try-except 无法捕获。
    # 放心跳过，后续的 TensorRT (trtexec) 会接管所有的计算图优化工作。
    print("🚀 开始导出 ONNX (已强制禁用 simplify)...")
    try:
        model.export(format="onnx", opset=13, simplify=False, imgsz=imgsz)
    except Exception as e:
        print(f"⚠️ 使用 opset 13 失败，尝试 opset 12: {e}")
        model.export(format="onnx", opset=12, simplify=False, imgsz=imgsz)

    if os.path.exists(onnx_path):
        print(f"✅ 已成功导出 ONNX: {onnx_path}")
    else:
        print("❌ ONNX 导出失败")

if __name__ == "__main__":
    pt_file = r"D:\Code\yolov13-aimbot\models\best256_yolo26.pt"
    onnx_file = pt_file.replace(".pt", ".onnx")
    engine_file = pt_file.replace(".pt", ".engine")

    # 1. 导出 ONNX
    export_to_onnx(pt_file, onnx_file, imgsz=256)

# trtexec --onnx=best256.onnx --saveEngine=best256.engine --fp16




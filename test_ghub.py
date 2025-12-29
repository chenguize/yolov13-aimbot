# test_ghub_import.py - 使用现有 output.py 中的 gHub 进行鼠标测试
# 运行方式：以管理员身份运行 python test_ghub_import.py
# 预期：鼠标向右上猛飞300像素，然后回来

import time
from output import gHub   # 直接导入你现有的 gHub 实例

print("\n=== GHUB 鼠标导入测试开始 ===")
print("请把鼠标放在屏幕中间，准备好被拉飞！")
print("3秒后开始测试...")
time.sleep(3)

print("测试1: 向右上飞 300, -300")
gHub.mouse_xy(300.5, -300.6)
time.sleep(0.8)

print("测试2: 回到原位 -300, 300")
gHub.mouse_xy(-300.68, 300.92)
time.sleep(0.5)

print("\n测试结束！")
print("如果鼠标明显动了（飞到右上再回来），说明 gHub 实例正常，DLL 能用！")
print("如果没动，请检查：")
print("1. 是否以管理员身份运行")
print("2. ghub_mouse.dll 是否在 output.py 同目录")
print("3. Logitech G HUB 是否运行中并更新到最新版")
print("4. Windows '增强指针精度' 是否关闭（设置 → 鼠标 → 其他选项 → 取消勾选）")
input("按回车退出...")
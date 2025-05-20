# -*- coding: utf-8 -*-

import pyautogui
import numpy as np
import mss
from PIL import Image
import gc  # 添加垃圾回收模块


def is_windowized(window_title: str):
    """
    判断目标是否窗口化
    """
    # 获取当前所有窗口的标题
    window_titles = [window.title for window in pyautogui.getAllWindows()]

    # 检查是否存在deltaforce窗口
    if window_title in window_titles:
        return True
    else:
        return False


def get_window_postion(target_app: str):
    """
    获取目标窗口的坐标
    """
    window_info = pyautogui.getWindowsWithTitle(target_app)[0]
    return [window_info.left, window_info.top, window_info.right, window_info.bottom]


def get_screenshot(debug_mode=False):
    """
    全屏截图函数，使用mss加速
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主屏
        img = sct.grab(monitor)
        screenshot = Image.frombytes("RGB", img.size, img.rgb)

        if debug_mode:
            screenshot.save("screenshot.png")

        # 返回前转换为numpy数组并手动释放PIL资源
        result = np.array(screenshot)
        screenshot.close()

        # 手动清理
        del img
        del screenshot
        gc.collect()

        return result


def get_windowshot(range: list, debug_mode=False):
    """优化的范围截图函数"""
    # 使用静态变量缓存mss实例
    if not hasattr(get_windowshot, "sct"):
        get_windowshot.sct = mss.mss()

    screen_size = pyautogui.size()
    if range[0] < 1:
        range = [
            int(screen_size.width * range[0]),
            int(screen_size.height * range[1]),
            int(screen_size.width * range[2]),
            int(screen_size.height * range[3]),
        ]
    monitor = {
        "left": range[0],
        "top": range[1],
        "width": range[2] - range[0],
        "height": range[3] - range[1],
    }

    # 直接获取numpy数组，避免PIL转换
    img = get_windowshot.sct.grab(monitor)
    img_np = np.array(img, dtype=np.uint8)

    # 仅BGR通道组合，跳过alpha通道
    img_np = img_np[:, :, :3]

    # 仅在调试模式下保存图像
    if debug_mode:
        Image.fromarray(img_np).save("screenshot.png")

    # 手动清理
    del monitor

    return img_np


def mouse_click(position: list, num: int = 1):
    """优化的鼠标点击函数"""
    x = position[0]
    y = position[1]
    if x < 1:
        screen_size = pyautogui.size()
        x = int(screen_size.width * x)
        y = int(screen_size.height * y)

    # 直接移动到目标位置
    pyautogui.moveTo(x, y, duration=0.01)  # 减少移动时间但保留少量防止游戏不识别

    # 快速点击
    for _ in range(num):
        pyautogui.click()


def get_mouse_position():
    """
    获取鼠标当前位置
    """
    return list(pyautogui.position())


def main():
    get_screenshot(debug_mode=True)


if __name__ == "__main__":
    main()

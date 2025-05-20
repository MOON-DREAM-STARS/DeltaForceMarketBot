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
    """
    范围截图函数，使用mss加速

    range：截图范围，[left, top, right, bottom]，支持比例
    """
    with mss.mss() as sct:
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
        del monitor
        gc.collect()

        return result


def mouse_click(positon: list, num: int = 1):
    """
    postion：鼠标点击位置，[x, y]

    num：点击次数，默认点击一次
    """
    x = positon[0]
    y = positon[1]
    if x < 1:
        screen_size = pyautogui.size()
        x = int(screen_size.width * x)
        y = int(screen_size.height * y)
    for i in range(num):
        pyautogui.moveTo(x, y)
        pyautogui.mouseDown()
        pyautogui.mouseUp()


def get_mouse_position():
    """
    获取鼠标当前位置
    """
    return list(pyautogui.position())


def main():
    get_screenshot(debug_mode=True)


if __name__ == "__main__":
    main()

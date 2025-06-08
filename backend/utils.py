# -*- coding: utf-8 -*-

import pyautogui
import numpy as np
import mss
from PIL import Image
import gc  # 添加垃圾回收模块
import win32gui
import win32ui
import win32con
import win32api


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


def get_screenshot_win32():
    """
    使用win32api进行截图
    """
    # 获取屏幕分辨率
    width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
    left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
    top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    # 创建设备环境
    hdesktop = win32gui.GetDesktopWindow()
    hwndDC = win32gui.GetWindowDC(hdesktop)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # 创建bitmap对象
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # 截图
    saveDC.BitBlt((0, 0), (width, height), mfcDC, (left, top), win32con.SRCCOPY)

    # 获取bitmap信息
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    # 转换为numpy数组
    img = np.frombuffer(bmpstr, dtype="uint8")
    img.shape = (height, width, 4)
    img = img[:, :, :3]  # 去掉alpha通道

    # 清理资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hdesktop, hwndDC)

    return img


def get_screenshot_mss():
    """
    使用mss进行截图
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主屏
        img = sct.grab(monitor)
        screenshot = Image.frombytes("RGB", img.size, img.rgb)

        # 返回前转换为numpy数组并手动释放PIL资源
        result = np.array(screenshot)
        screenshot.close()

        # 手动清理
        del img
        del screenshot
        gc.collect()

        return result

def get_screenshot_mss_debug_monitors():
    """
    调试MSS：为所有检测到的监视器截图。
    """
    problem_found = False
    with mss.mss() as sct:
        for i, monitor in enumerate(sct.monitors):
            try:
                # 为每个监视器截图
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
                
                # 保存截图，文件名包含监视器索引和尺寸
                filename = f"debug_mss_monitor_{i}_left{monitor['left']}_top{monitor['top']}_w{monitor['width']}_h{monitor['height']}.png"
                img.save(filename)
                print(f"已保存截图: {filename}")

                # 简单检查是否全黑 (可选)
                img_np = np.array(img)
                if np.mean(img_np) < 5: # 阈值可以调整
                    print(f"警告: 监视器 {i} 的截图几乎全黑。")
                    problem_found = True

            except Exception as e:
                print(f"为监视器 {i} 截图失败: {e}")
                problem_found = True
    if not problem_found:
        print("所有监视器截图（可能）成功。请检查生成的图片。")
    return problem_found

def get_screenshot(method="mss", debug_mode=False):
    """
    全屏截图函数，根据method参数选择截图方法
    method: "mss" 或 "win32"
    """
    if method == "mss":
        result = get_screenshot_mss()
    elif method == "win32":
        result = get_screenshot_win32()
    else:
        raise ValueError(f"不支持的截图方法: {method}，仅支持 'mss' 或 'win32'")

    if debug_mode:
        Image.fromarray(result).save(f"screenshot_{method}.png")

    return result


def get_windowshot_win32(range: list):
    """
    使用win32api进行范围截图
    """
    screen_size = pyautogui.size()
    if range[0] < 1:
        range = [
            int(screen_size.width * range[0]),
            int(screen_size.height * range[1]),
            int(screen_size.width * range[2]),
            int(screen_size.height * range[3]),
        ]

    # 获取指定区域截图
    left, top, right, bottom = range
    width = right - left
    height = bottom - top

    hdesktop = win32gui.GetDesktopWindow()
    hwndDC = win32gui.GetWindowDC(hdesktop)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # 截取指定区域
    saveDC.BitBlt((0, 0), (width, height), mfcDC, (left, top), win32con.SRCCOPY)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    img = np.frombuffer(bmpstr, dtype="uint8")
    img.shape = (height, width, 4)
    img = img[:, :, :3]  # 去掉alpha通道

    # 清理资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hdesktop, hwndDC)

    return img


def get_windowshot_mss(range: list):
    """
    使用mss进行范围截图
    """
    # 使用静态变量缓存mss实例
    if not hasattr(get_windowshot_mss, "sct"):
        get_windowshot_mss.sct = mss.mss()

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
    img = get_windowshot_mss.sct.grab(monitor)
    img_np = np.array(img, dtype=np.uint8)

    # 仅BGR通道组合，跳过alpha通道
    img_np = img_np[:, :, :3]

    # 手动清理
    del monitor
    return img_np


def get_windowshot(range: list, method="mss", debug_mode=False):
    """
    范围截图函数，根据method参数选择截图方法
    method: "mss" 或 "win32"
    """
    if method == "mss":
        result = get_windowshot_mss(range)
    elif method == "win32":
        result = get_windowshot_win32(range)
    else:
        raise ValueError(f"不支持的截图方法: {method}，仅支持 'mss' 或 'win32'")

    if debug_mode:
        Image.fromarray(result).save(f"screenshot_{method}.png")

    return result


def mouse_click(position: list, num: int = 1):
    """优化的鼠标点击函数"""
    x = position[0]
    y = position[1]
    if x < 1:
        screen_size = pyautogui.size()
        x = int(screen_size.width * x)
        y = int(screen_size.height * y)

    # 直接移动到目标位置
    pyautogui.moveTo(x, y, duration=0.01)

    # 快速点击
    for _ in range(num):
        pyautogui.click()


def get_mouse_position():
    """
    获取鼠标当前位置
    """
    return list(pyautogui.position())


def main():
    # 测试两种截图方法
    print("测试MSS截图...")
    get_screenshot(method="mss", debug_mode=True)
    print("测试Win32截图...")
    get_screenshot(method="win32", debug_mode=True)


if __name__ == "__main__":
    # main()
    get_screenshot_mss_debug_monitors()

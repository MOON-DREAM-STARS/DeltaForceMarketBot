import sys
import gc  # 添加垃圾回收模块
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QThread
from GUI.AppGUI import Ui_MainWindow
from backend.BuyBot import BuyBot
from backend.utils import *
import keyboard


class KeyMonitor(QObject):
    key_pressed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        keyboard.on_press(self.handle_key)

    def handle_key(self, event):
        if event.name == "f8":
            self.key_pressed.emit(0)
            print("开始循环")
        elif event.name == "f9":
            self.key_pressed.emit(1)
            print("停止循环")


class Worker(QThread):
    update_signal = pyqtSignal(int)
    param_update = pyqtSignal(int)  # 新增参数更新信号

    def __init__(self, buybot):
        super().__init__()
        self.buybot = buybot
        self._is_running = False
        self.lock = QtCore.QMutex()
        self.ideal_price = 0
        self.unacceptable_price = 0
        self.loop_gap = 0
        self.is_convertible = True
        self.is_key_mode = False
        self.mouse_position = []
        self.mouse_position_lock = QtCore.QMutex()
        self.param_lock = QtCore.QMutex()  # 参数专用锁

    def record_mouse_position(self):
        """记录鼠标位置"""
        self.mouse_position_lock.lock()
        self.mouse_position = get_mouse_position()
        self.mouse_position_lock.unlock()

    def run(self):
        # 添加状态变量，跟踪是否在商品页面
        in_product_page = False

        while True:
            # 获取运行状态
            self.lock.lock()
            running = self._is_running
            self.lock.unlock()

            if running:
                try:
                    # 获取当前参数值
                    self.param_lock.lock()
                    current_ideal = self.ideal_price
                    current_unacceptable = self.unacceptable_price
                    current_convertible = self.is_convertible
                    current_key_mode = self.is_key_mode
                    self.param_lock.unlock()

                    # 仅在需要时进入商品页面
                    if not in_product_page:
                        mouse_click(self.mouse_position, num=1)
                        in_product_page = True

                    # 检测逻辑
                    lowest_price = self.buybot.detect_price(
                        is_convertible=current_convertible, debug_mode=False
                    )
                    self.update_signal.emit(lowest_price)

                    if current_key_mode:
                        # 钥匙卡模式
                        if lowest_price > current_ideal:
                            print(
                                "当前价格：",
                                lowest_price,
                                "高于理想价格",
                                current_ideal,
                                "，免费刷新价格",
                            )
                            self.buybot.freerefresh(good_postion=self.mouse_position)
                            # 已经重新进入商品页面，保持in_product_page=True
                        else:
                            print(
                                "当前价格：",
                                lowest_price,
                                "低于理想价格",
                                current_ideal,
                                "，购买一张后循环结束",
                            )
                            self.buybot.refresh(is_convertible=False)
                            self.set_running(False)
                            print("停止循环")
                            in_product_page = False  # 循环停止，重置状态
                    else:
                        # 正常模式
                        if lowest_price > current_unacceptable:
                            print(
                                "当前价格：",
                                lowest_price,
                                "高于最高价格",
                                current_unacceptable,
                                "，免费刷新价格",
                            )
                            self.buybot.freerefresh(good_postion=self.mouse_position)
                            # 已经重新进入商品页面，保持in_product_page=True
                        elif (
                            lowest_price <= current_unacceptable
                            and lowest_price > current_ideal
                        ):
                            print(
                                "当前价格：",
                                lowest_price,
                                "低于最高价格",
                                current_unacceptable,
                                "高于理想价格",
                                current_ideal,
                                "，刷新价格",
                            )
                            self.buybot.refresh(is_convertible=current_convertible)
                            # 刷新后仍在商品页面，保持in_product_page=True
                        elif lowest_price <= current_ideal:
                            print(
                                "当前价格：",
                                lowest_price,
                                "低于理想价格",
                                current_ideal,
                                "，开始购买",
                            )
                            self.buybot.buy(is_convertible=current_convertible)
                            # 购买后仍在商品页面，保持in_product_page=True

                    # 周期性强制垃圾回收
                    gc.collect()

                except Exception as e:
                    print(f"操作失败: {str(e)}")
                    in_product_page = False  # 出错时重置状态，下次重新进入
                self.msleep(self.loop_gap)
            else:
                in_product_page = False  # 不运行时重置状态
                self.msleep(100)

    def update_params(self, ideal, unacceptable, convertible, key_mode, loop_gap):
        """线程安全更新参数"""
        self.param_lock.lock()
        self.ideal_price = ideal
        self.unacceptable_price = unacceptable
        self.loop_gap = loop_gap
        self.is_convertible = convertible
        self.is_key_mode = key_mode
        self.param_lock.unlock()

    def set_running(self, state):
        """线程安全更新运行状态"""
        self.lock.lock()
        self._is_running = state
        self.lock.unlock()


def runApp():
    app = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    mainWindow = Ui_MainWindow()
    mainWindow.setupUi(window)

    # 初始化输入部分
    mainWindow.textEdit_ideal_price.setText("0")
    mainWindow.textEdit_unacceptable_price.setText("0")
    mainWindow.textEdit_loop_gap.setText("150")
    mainWindow.is_convertiable.setChecked(True)
    mainWindow.is_key_mode.setChecked(False)

    # 创建监控线程
    key_monitor = KeyMonitor()
    worker = Worker(BuyBot(ocr_engine="paddleocr"))  # 改为paddleocr

    # 信号连接
    def handle_key_event(x):
        if x == 0:
            worker.record_mouse_position()
        worker.set_running(x == 0)

    key_monitor.key_pressed.connect(handle_key_event)

    def handle_text_change():
        try:
            ideal = int(mainWindow.textEdit_ideal_price.toPlainText())
            unaccept = int(mainWindow.textEdit_unacceptable_price.toPlainText())
            loop_gap = int(mainWindow.textEdit_loop_gap.toPlainText())
            is_convertible = mainWindow.is_convertiable.isChecked()
            is_key_mode = mainWindow.is_key_mode.isChecked()
            worker.update_params(ideal, unaccept, is_convertible, is_key_mode, loop_gap)
            mainWindow.label_lowest_price_value.setStyleSheet("color: black;")
        except ValueError:
            mainWindow.label_lowest_price_value.setStyleSheet("color: red;")

    # 确保两个输入框都连接
    mainWindow.textEdit_ideal_price.textChanged.connect(handle_text_change)
    mainWindow.textEdit_unacceptable_price.textChanged.connect(handle_text_change)
    mainWindow.textEdit_loop_gap.textChanged.connect(handle_text_change)
    mainWindow.is_convertiable.stateChanged.connect(handle_text_change)
    mainWindow.is_key_mode.stateChanged.connect(handle_text_change)

    window.show()
    worker.start()
    app.exec_()


def main():
    return runApp()


if __name__ == "__main__":
    print("正在初始化")
    sys.exit(main())

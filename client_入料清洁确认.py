# -*- coding: UTF-8 -*-

"""
@User:         smile
@Author:       Smile
@Email:        Xiaofei.Smile365@Gmail.com
@Date Time:    2022/2/15 10:15
@IDE:          PyCharm
@Source  : python3 -m pip install *** -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import os
import sys

if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']  # PyQt自身存在bug，打包时环境变量出错，无法运行，此语句对环境变量进行重新配置，消除bug

from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

from socket import *


# noinspection PyAttributeOutsideInit
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)

        self.screen = QDesktopWidget().screenGeometry()

        self.setWindowIcon(QIcon('./src/platform.ico'))
        self.setWindowTitle("AOI 抛料率异常处理反馈程式")

        self.size = self.geometry()
        self.move((self.screen.width() - self.size.width()) / 4, (self.screen.height() - self.size.height()) / 4)

        self.status = self.statusBar()

        self.setFixedSize(1024, 384)

        self.set_label()
        self.set_layout()

        self.main_frame = QWidget()
        self.main_frame.setLayout(self.layout_v_windows)
        self.setCentralWidget(self.main_frame)

        self.status.showMessage("AOI 抛料率异常处理反馈程式", 2000)

        # 界面设置初始化
        self.label_title.setFocus()  # 取消界面自动获取焦点

        server_ip, ok = QInputDialog.getText(self, '请输入Server IP', '输入 Server IP:')
        if ok:
            self.server_ip = str(server_ip)

            try:
                self.client = socket()
                ip_port = (self.server_ip, 8888)
                self.client.connect(ip_port)
            except:
                QMessageBox.about(self, '警告', 'Server IP 无法连接，请联系 苏晓飞（8610-2484/8690-0070）')
                sys.exit(0)

        else:
            sys.exit(0)

        self.line_clean_panel_check_total_sum.setEnabled(False)
        self.line_clean_panel_check_ok_sum.setEnabled(False)
        self.line_clean_panel_check_ng_sum.setEnabled(False)
        self.button_clean_panel_check_send.setEnabled(False)

    def set_label(self):
        # 界面标题
        self.label_title = QLabel()
        self.label_title.setText("<b>AOI 抛料率异常处理反馈程式<b>")
        self.label_title.setFont(QFont("SanSerif", 24))
        self.label_title.setStyleSheet("Color: RGB(64, 224, 208)")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setFixedSize(1024, 50)
        self.layout_h_label_title = QHBoxLayout()
        self.layout_h_label_title.addWidget(self.label_title)

        # 清洁手法
        self.clean_panel_check_title = QLabel()
        self.clean_panel_check_title.setText("<b>入料清洁确认<b>")
        self.clean_panel_check_title.setFont(QFont("SanSerif", 20))
        self.clean_panel_check_title.setStyleSheet("Color: RGB(64, 224, 208)")
        self.clean_panel_check_title.setAlignment(Qt.AlignCenter)
        self.clean_panel_check_title.setFixedSize(200, 30)
        self.layout_h_clean_panel_check_title = QHBoxLayout()
        self.layout_h_clean_panel_check_title.addWidget(self.clean_panel_check_title)

        self.clean_panel_study = QLabel()
        self.clean_panel_study.setText("<b>人员清洁手法现场Check - Done<b>")
        self.clean_panel_study.setFont(QFont("SanSerif", 20))
        self.clean_panel_study.setStyleSheet("Color: RGB(64, 224, 208)")
        self.clean_panel_study.setAlignment(Qt.AlignCenter)
        self.clean_panel_study.setFixedSize(463, 30)
        self.layout_h_clean_panel_study = QHBoxLayout()
        self.layout_h_clean_panel_study.addWidget(self.clean_panel_study)

        self.button_clean_panel_study_send = QPushButton()
        self.button_clean_panel_study_send.setText("发送")
        self.button_clean_panel_study_send.clicked.connect(self.clean_panel_study_send)
        self.button_clean_panel_study_send.setToolTip("向异常处理控制平台发送反馈信息。")
        self.button_clean_panel_study_send.setFont(QFont("微软雅黑", 10))
        self.button_clean_panel_study_send.setFixedSize(100, 50)
        self.layout_button_clean_panel_study_send = QHBoxLayout()
        self.layout_button_clean_panel_study_send.addWidget(self.button_clean_panel_study_send)

        self.layout_h_clean_panel_study_send = QHBoxLayout()
        self.layout_h_clean_panel_study_send.addStretch(3)
        self.layout_h_clean_panel_study_send.addLayout(self.layout_h_clean_panel_study)
        self.layout_h_clean_panel_study_send.addStretch(3)
        self.layout_h_clean_panel_study_send.addLayout(self.layout_button_clean_panel_study_send)
        self.layout_h_clean_panel_study_send.addStretch(3)

        self.clean_panel_check_total_sum = QLabel()
        self.clean_panel_check_total_sum.setFocus()
        self.clean_panel_check_total_sum.setText("<b>抽检数量<b>")
        self.clean_panel_check_total_sum.setFont(QFont("SanSerif", 16))
        self.clean_panel_check_total_sum.setStyleSheet("Color: RGB(64, 224, 208)")
        self.clean_panel_check_total_sum.setAlignment(Qt.AlignCenter)
        self.clean_panel_check_total_sum.setFixedSize(150, 30)
        self.layout_h_clean_panel_check_total_sum = QHBoxLayout()
        self.layout_h_clean_panel_check_total_sum.addWidget(self.clean_panel_check_total_sum)

        self.line_clean_panel_check_total_sum = QLineEdit()
        self.line_clean_panel_check_total_sum.setAlignment(Qt.AlignLeft)
        self.line_clean_panel_check_total_sum.setPlaceholderText("请输入抽检数量...")
        self.line_clean_panel_check_total_sum.setFont(QFont("微软雅黑", 8))
        self.line_clean_panel_check_total_sum.setFixedSize(100, 30)
        self.layout_h_line_clean_panel_check_total_sum = QHBoxLayout()
        self.layout_h_line_clean_panel_check_total_sum.addWidget(self.line_clean_panel_check_total_sum)

        self.layout_v_clean_panel_check_total_sum = QVBoxLayout()
        self.layout_v_clean_panel_check_total_sum.addLayout(self.layout_h_clean_panel_check_total_sum)
        self.layout_v_clean_panel_check_total_sum.addLayout(self.layout_h_line_clean_panel_check_total_sum)

        self.clean_panel_check_ok_sum = QLabel()
        self.clean_panel_check_ok_sum.setFocus()
        self.clean_panel_check_ok_sum.setText("<b>OK 数量<b>")
        self.clean_panel_check_ok_sum.setFont(QFont("SanSerif", 16))
        self.clean_panel_check_ok_sum.setStyleSheet("Color: RGB(64, 224, 208)")
        self.clean_panel_check_ok_sum.setAlignment(Qt.AlignCenter)
        self.clean_panel_check_ok_sum.setFixedSize(150, 30)
        self.layout_h_clean_panel_check_ok_sum = QHBoxLayout()
        self.layout_h_clean_panel_check_ok_sum.addWidget(self.clean_panel_check_ok_sum)

        self.line_clean_panel_check_ok_sum = QLineEdit()
        self.line_clean_panel_check_ok_sum.setAlignment(Qt.AlignLeft)
        self.line_clean_panel_check_ok_sum.setPlaceholderText("请输入 OK 数量...")
        self.line_clean_panel_check_ok_sum.setFont(QFont("微软雅黑", 8))
        self.line_clean_panel_check_ok_sum.setFixedSize(100, 30)
        self.layout_h_line_clean_panel_check_ok_sum = QHBoxLayout()
        self.layout_h_line_clean_panel_check_ok_sum.addWidget(self.line_clean_panel_check_ok_sum)

        self.layout_v_clean_panel_check_ok_sum = QVBoxLayout()
        self.layout_v_clean_panel_check_ok_sum.addLayout(self.layout_h_clean_panel_check_ok_sum)
        self.layout_v_clean_panel_check_ok_sum.addLayout(self.layout_h_line_clean_panel_check_ok_sum)

        self.clean_panel_check_ng_sum = QLabel()
        self.clean_panel_check_ng_sum.setFocus()
        self.clean_panel_check_ng_sum.setText("<b>NG 数量<b>")
        self.clean_panel_check_ng_sum.setFont(QFont("SanSerif", 16))
        self.clean_panel_check_ng_sum.setStyleSheet("Color: RGB(64, 224, 208)")
        self.clean_panel_check_ng_sum.setAlignment(Qt.AlignCenter)
        self.clean_panel_check_ng_sum.setFixedSize(150, 30)
        self.layout_h_clean_panel_check_ng_sum = QHBoxLayout()
        self.layout_h_clean_panel_check_ng_sum.addWidget(self.clean_panel_check_ng_sum)

        self.line_clean_panel_check_ng_sum = QLineEdit()
        self.line_clean_panel_check_ng_sum.setAlignment(Qt.AlignLeft)
        self.line_clean_panel_check_ng_sum.setPlaceholderText("请输入 NG 数量...")
        self.line_clean_panel_check_ng_sum.setFont(QFont("微软雅黑", 8))
        self.line_clean_panel_check_ng_sum.setFixedSize(100, 30)
        self.layout_h_line_clean_panel_check_ng_sum = QHBoxLayout()
        self.layout_h_line_clean_panel_check_ng_sum.addWidget(self.line_clean_panel_check_ng_sum)

        self.layout_v_clean_panel_check_ng_sum = QVBoxLayout()
        self.layout_v_clean_panel_check_ng_sum.addLayout(self.layout_h_clean_panel_check_ng_sum)
        self.layout_v_clean_panel_check_ng_sum.addLayout(self.layout_h_line_clean_panel_check_ng_sum)

        self.layout_h_clean_panel_check_num = QHBoxLayout()
        self.layout_h_clean_panel_check_num.addLayout(self.layout_v_clean_panel_check_total_sum)
        self.layout_h_clean_panel_check_num.addLayout(self.layout_v_clean_panel_check_ok_sum)
        self.layout_h_clean_panel_check_num.addLayout(self.layout_v_clean_panel_check_ng_sum)

        self.button_clean_panel_check_send = QPushButton()
        self.button_clean_panel_check_send.setText("发送")
        self.button_clean_panel_check_send.clicked.connect(self.clean_panel_check_send)
        self.button_clean_panel_check_send.setToolTip("向异常处理控制平台发送反馈信息。")
        self.button_clean_panel_check_send.setFont(QFont("微软雅黑", 10))
        self.button_clean_panel_check_send.setFixedSize(100, 50)
        self.layout_button_clean_panel_check_send = QHBoxLayout()
        self.layout_button_clean_panel_check_send.addWidget(self.button_clean_panel_check_send)

        self.layout_h_clean_panel_check_send = QHBoxLayout()
        self.layout_h_clean_panel_check_send.addLayout(self.layout_h_clean_panel_check_num)
        self.layout_h_clean_panel_check_send.addLayout(self.layout_button_clean_panel_check_send)

        self.layout_v_clean_panel_check = QVBoxLayout()
        self.layout_v_clean_panel_check.addLayout(self.layout_h_clean_panel_study_send)
        self.layout_v_clean_panel_check.addLayout(self.layout_h_clean_panel_check_send)

    def set_layout(self):
        self.layout_h_title = QHBoxLayout()
        self.layout_h_title.addLayout(self.layout_h_label_title)

        self.layout_h_clean_panel_check = QHBoxLayout()
        self.layout_h_clean_panel_check.addStretch(1)
        self.layout_h_clean_panel_check.addLayout(self.layout_h_clean_panel_check_title)
        self.layout_h_clean_panel_check.addStretch(1)
        self.layout_h_clean_panel_check.addLayout(self.layout_v_clean_panel_check)
        self.layout_h_clean_panel_check.addStretch(2)

        self.layout_v_windows = QVBoxLayout()
        self.layout_v_windows.addLayout(self.layout_h_title)
        self.layout_v_windows.addStretch(1)
        self.layout_v_windows.addLayout(self.layout_h_clean_panel_check)
        self.layout_v_windows.addStretch(2)

    def clean_panel_study_send(self):
        """
        编写通讯代码并将数据发送
        """
        data = [{'name': 'clean_panel_study', 'schedule': 'done', 'total_num': 0,
                 'ok_num': 0, 'ng_num': 0}]
        try:
            self.client.send(str(data).encode("utf-8"))

            self.line_clean_panel_check_total_sum.setEnabled(True)
            self.line_clean_panel_check_ok_sum.setEnabled(True)
            self.line_clean_panel_check_ng_sum.setEnabled(True)
            self.button_clean_panel_check_send.setEnabled(True)

            self.button_clean_panel_study_send.setEnabled(False)
        except:
            QMessageBox.about(self, '警告', 'Server IP 无法连接，请联系 苏晓飞（8610-2484/8690-0070）')

    def clean_panel_check_send(self):
        if self.line_clean_panel_check_total_sum.text() != '' and self.line_clean_panel_check_ok_sum.text() != '' and self.line_clean_panel_check_ng_sum.text() != '':
            if self.line_clean_panel_check_total_sum.text().isdigit() and self.line_clean_panel_check_ok_sum.text().isdigit() and self.line_clean_panel_check_ng_sum.text().isdigit():
                self.clean_panel_check_total_sum = int(self.line_clean_panel_check_total_sum.text())
                self.clean_panel_check_ok_sum = int(self.line_clean_panel_check_ok_sum.text())
                self.clean_panel_check_ng_sum = int(self.line_clean_panel_check_ng_sum.text())

                """
                编写通讯代码并将数据发送
                """
                data = [{'name': 'clean_panel_check', 'schedule': 'done', 'total_num': self.clean_panel_check_total_sum,
                         'ok_num': self.clean_panel_check_ok_sum, 'ng_num': self.clean_panel_check_ng_sum}]
                try:
                    self.client.send(str(data).encode("utf-8"))

                    self.line_clean_panel_check_total_sum.setEnabled(False)
                    self.line_clean_panel_check_ok_sum.setEnabled(False)
                    self.line_clean_panel_check_ng_sum.setEnabled(False)
                    self.button_clean_panel_check_send.setEnabled(False)
                except:
                    QMessageBox.about(self, '警告', 'Server IP 无法连接，请联系 苏晓飞（8610-2484/8690-0070）')

            else:
                QMessageBox.about(self, '警告', '输入数据必须为整数！')
        else:
            QMessageBox.about(self, '警告', '请输入相关数据！')


if __name__ == '__main__':
    app_system = QApplication(sys.argv)
    form_system = MainWindow()
    form_system.show()
    sys.exit(app_system.exec_())

    pass

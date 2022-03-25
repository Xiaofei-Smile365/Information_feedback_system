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

import datetime
import threading

from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import *
from PyQt5.Qt import QThread

from socket import *

import sip
import csv
from ftplib import FTP

global msg_from_client


# 继承QThread
# noinspection PyAttributeOutsideInit
class Thread_server(QThread):
    def __init__(self, server_ip):
        super().__init__()
        self.server_ip = server_ip

    def run(self):
        self.server = socket()
        self.ip_port = (self.server_ip, 8888)
        self.server.bind(self.ip_port)
        self.server.listen(10)
        while True:
            self.conn, self.client_addr = self.server.accept()
            thread_talk = threading.Thread(target=self.talk, args=(self.conn, self.client_addr))
            thread_talk.start()

    def talk(self, connect, addr):
        global msg_from_client
        while True:
            try:
                self.client_from_msg = connect.recv(1024)
                if str(self.client_from_msg) != "b''":
                    # print("来自客户端%s端口%s的消息: " % (addr[0], addr[1]), self.client_from_msg)
                    msg_from_client = {'client': addr[0], 'port': addr[1], 'msg': self.client_from_msg}
                if not self.client_from_msg:
                    break
            except:
                pass


# noinspection PyAttributeOutsideInit
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)

        self.screen = QDesktopWidget().screenGeometry()

        self.setWindowIcon(QIcon('./src/platform.ico'))
        self.setWindowTitle("AOI 抛料率异常处理管控平台")

        self.size = self.geometry()
        self.move((self.screen.width() - self.size.width()) / 4, (self.screen.height() - self.size.height()) / 4)

        self.status = self.statusBar()

        self.setFixedSize(1366, 868)

        self.set_label()
        self.set_layout()

        self.main_frame = QWidget()
        self.main_frame.setLayout(self.layout_v_windows)
        self.setCentralWidget(self.main_frame)

        self.status.showMessage("AOI 抛料率异常处理管控平台", 2000)

        # 界面设置初始化
        self.label_title.setFocus()  # 取消界面自动获取焦点

        server_ip, ok = QInputDialog.getText(self, '请输入Server IP', '输入 Server IP:')
        if ok:
            self.server_ip = str(server_ip)
            try:
                """
                Socket Server code
                """
                self.thread_socket_server = Thread_server(self.server_ip)  # 创建线程
                self.thread_socket_server.start()  # 开始线程
            except:
                QMessageBox.about(self, '警告', 'Server for Socket 无法建立，请联系 苏晓飞（8610-2484/8690-0070）')
                sys.exit(0)

        else:
            sys.exit(0)

    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key_1:
    #         self.change_step1()
    #     if event.key() == Qt.Key_2:
    #         self.change_step2()
    #     if event.key() == Qt.Key_3:
    #         self.change_step3()
    #     if event.key() == Qt.Key_4:
    #         self.change_step4()
    #     if event.key() == Qt.Key_5:
    #         self.change_step5()

    def set_label(self):
        # 界面标题
        self.label_title = QLabel()
        self.label_title.setText("<b>AOI 抛料率异常处理管控平台<b>")
        self.label_title.setFont(QFont("SanSerif", 24))
        self.label_title.setStyleSheet("Color: RGB(64, 224, 208)")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setFixedSize(1024, 50)
        self.layout_h_label_title = QHBoxLayout()
        self.layout_h_label_title.addWidget(self.label_title)

        self.button_start = QPushButton()
        self.button_start.setText("启动")
        self.button_start.clicked.connect(self.time_start)
        self.button_start.setToolTip("启动异常处理管控并倒计时15min。")
        self.button_start.setFont(QFont("微软雅黑", 20))
        self.button_start.setFixedSize(200, 100)
        self.layout_button_start = QHBoxLayout()
        self.layout_button_start.addWidget(self.button_start)

        # 来料洁净
        self.source_panel_check_title = QLabel()
        self.source_panel_check_title.setFocus()
        self.source_panel_check_title.setText("<b>PSE(制程)-来料洁净确认<b>")
        self.source_panel_check_title.setFont(QFont("SanSerif", 20))
        self.source_panel_check_title.setStyleSheet("Color: RGB(64, 224, 208)")
        self.source_panel_check_title.setAlignment(Qt.AlignCenter)
        self.source_panel_check_title.setFixedSize(315, 30)
        self.layout_h_source_panel_check_title = QHBoxLayout()
        self.layout_h_source_panel_check_title.addWidget(self.source_panel_check_title)

        self.line_source_panel_check_sum = QLineEdit()
        self.line_source_panel_check_sum.setAlignment(Qt.AlignLeft)
        self.line_source_panel_check_sum.setPlaceholderText("来料洁净抽检数据待更新......")
        self.line_source_panel_check_sum.setFont(QFont("微软雅黑", 8))
        self.line_source_panel_check_sum.setFixedSize(806, 30)
        self.line_source_panel_check_sum.setEnabled(False)
        self.layout_h_line_source_panel_check_sum = QHBoxLayout()
        self.layout_h_line_source_panel_check_sum.addWidget(self.line_source_panel_check_sum)

        self.label_photo_alarm_source_panel_check = QLabel()
        self.label_photo_alarm_source_panel_check.setAlignment(Qt.AlignCenter)
        self.label_photo_alarm_source_panel_check.setToolTip("红绿灯")
        self.label_photo_alarm_source_panel_check.setPixmap(QPixmap("./src/gray.png").scaled(80, 80))
        self.label_photo_alarm_source_panel_check.setFixedSize(80, 80)
        self.layout_h_label_photo_alarm_source_panel_check = QHBoxLayout()
        self.layout_h_label_photo_alarm_source_panel_check.addWidget(self.label_photo_alarm_source_panel_check)

        self.layout_h_source_panel_check = QHBoxLayout()
        self.layout_h_source_panel_check.addStretch(3)
        self.layout_h_source_panel_check.addLayout(self.layout_h_source_panel_check_title)
        self.layout_h_source_panel_check.addStretch(1)
        self.layout_h_source_panel_check.addLayout(self.layout_h_line_source_panel_check_sum)
        self.layout_h_source_panel_check.addStretch(1)
        self.layout_h_source_panel_check.addLayout(self.layout_h_label_photo_alarm_source_panel_check)
        self.layout_h_source_panel_check.addStretch(3)

        # 入料清洁
        self.clean_panel_check_title = QLabel()
        self.clean_panel_check_title.setFocus()
        self.clean_panel_check_title.setText("<b>MFG(制造)-入料清洁确认<b>")
        self.clean_panel_check_title.setFont(QFont("SanSerif", 20))
        self.clean_panel_check_title.setStyleSheet("Color: RGB(64, 224, 208)")
        self.clean_panel_check_title.setAlignment(Qt.AlignCenter)
        self.clean_panel_check_title.setFixedSize(315, 30)
        self.layout_h_clean_panel_check_title = QHBoxLayout()
        self.layout_h_clean_panel_check_title.addWidget(self.clean_panel_check_title)

        self.line_clean_panel_study_sum = QLineEdit()
        self.line_clean_panel_study_sum.setAlignment(Qt.AlignLeft)
        self.line_clean_panel_study_sum.setPlaceholderText("清洁手法check进度待更新......")
        self.line_clean_panel_study_sum.setFont(QFont("微软雅黑", 8))
        self.line_clean_panel_study_sum.setFixedSize(400, 30)
        self.line_clean_panel_study_sum.setEnabled(False)
        self.layout_h_line_clean_panel_study_sum = QHBoxLayout()
        self.layout_h_line_clean_panel_study_sum.addWidget(self.line_clean_panel_study_sum)

        self.line_clean_panel_check_sum = QLineEdit()
        self.line_clean_panel_check_sum.setAlignment(Qt.AlignLeft)
        self.line_clean_panel_check_sum.setPlaceholderText("入料清洁抽检数据待更新......")
        self.line_clean_panel_check_sum.setFont(QFont("微软雅黑", 8))
        self.line_clean_panel_check_sum.setFixedSize(400, 30)
        self.line_clean_panel_check_sum.setEnabled(False)
        self.layout_h_line_clean_panel_check_sum = QHBoxLayout()
        self.layout_h_line_clean_panel_check_sum.addWidget(self.line_clean_panel_check_sum)

        self.label_photo_alarm_clean_panel_check = QLabel()
        self.label_photo_alarm_clean_panel_check.setAlignment(Qt.AlignCenter)
        self.label_photo_alarm_clean_panel_check.setToolTip("红绿灯")
        self.label_photo_alarm_clean_panel_check.setPixmap(QPixmap("./src/gray.png").scaled(80, 80))
        self.label_photo_alarm_clean_panel_check.setFixedSize(80, 80)
        self.layout_h_label_photo_alarm_clean_panel_check = QHBoxLayout()
        self.layout_h_label_photo_alarm_clean_panel_check.addWidget(self.label_photo_alarm_clean_panel_check)

        self.layout_h_clean_panel_check = QHBoxLayout()
        self.layout_h_clean_panel_check.addStretch(3)
        self.layout_h_clean_panel_check.addLayout(self.layout_h_clean_panel_check_title)
        self.layout_h_clean_panel_check.addStretch(1)
        self.layout_h_clean_panel_check.addLayout(self.layout_h_line_clean_panel_study_sum)
        self.layout_h_clean_panel_check.addLayout(self.layout_h_line_clean_panel_check_sum)
        self.layout_h_clean_panel_check.addStretch(1)
        self.layout_h_clean_panel_check.addLayout(self.layout_h_label_photo_alarm_clean_panel_check)
        self.layout_h_clean_panel_check.addStretch(3)

        # 机台参数
        self.eqpt_para_check_title = QLabel()
        self.eqpt_para_check_title.setFocus()
        self.eqpt_para_check_title.setText("<b>EQ (设备)-机台参数优化<b>")
        self.eqpt_para_check_title.setFont(QFont("SanSerif", 20))
        self.eqpt_para_check_title.setStyleSheet("Color: RGB(64, 224, 208)")
        self.eqpt_para_check_title.setAlignment(Qt.AlignCenter)
        self.eqpt_para_check_title.setFixedSize(315, 30)
        self.layout_h_eqpt_para_check_title = QHBoxLayout()
        self.layout_h_eqpt_para_check_title.addWidget(self.eqpt_para_check_title)

        self.line_eqpt_para_study_sum = QLineEdit()
        self.line_eqpt_para_study_sum.setAlignment(Qt.AlignLeft)
        self.line_eqpt_para_study_sum.setPlaceholderText("机台参数优化进度待更新......")
        self.line_eqpt_para_study_sum.setFont(QFont("微软雅黑", 8))
        self.line_eqpt_para_study_sum.setFixedSize(400, 30)
        self.line_eqpt_para_study_sum.setEnabled(False)
        self.layout_h_line_eqpt_para_study_sum = QHBoxLayout()
        self.layout_h_line_eqpt_para_study_sum.addWidget(self.line_eqpt_para_study_sum)

        self.line_eqpt_para_check_sum = QLineEdit()
        self.line_eqpt_para_check_sum.setAlignment(Qt.AlignLeft)
        self.line_eqpt_para_check_sum.setPlaceholderText("机台参数优化后试跑数据待更新......")
        self.line_eqpt_para_check_sum.setFont(QFont("微软雅黑", 8))
        self.line_eqpt_para_check_sum.setFixedSize(400, 30)
        self.line_eqpt_para_check_sum.setEnabled(False)
        self.layout_h_line_eqpt_para_check_sum = QHBoxLayout()
        self.layout_h_line_eqpt_para_check_sum.addWidget(self.line_eqpt_para_check_sum)

        self.label_photo_alarm_eqpt_para_check = QLabel()
        self.label_photo_alarm_eqpt_para_check.setAlignment(Qt.AlignCenter)
        self.label_photo_alarm_eqpt_para_check.setToolTip("红绿灯")
        self.label_photo_alarm_eqpt_para_check.setPixmap(QPixmap("./src/gray.png").scaled(80, 80))
        self.label_photo_alarm_eqpt_para_check.setFixedSize(80, 80)
        self.layout_h_label_photo_alarm_eqpt_para_check = QHBoxLayout()
        self.layout_h_label_photo_alarm_eqpt_para_check.addWidget(self.label_photo_alarm_eqpt_para_check)

        self.layout_h_eqpt_para_check = QHBoxLayout()
        self.layout_h_eqpt_para_check.addStretch(3)
        self.layout_h_eqpt_para_check.addLayout(self.layout_h_eqpt_para_check_title)
        self.layout_h_eqpt_para_check.addStretch(1)
        self.layout_h_eqpt_para_check.addLayout(self.layout_h_line_eqpt_para_study_sum)
        self.layout_h_eqpt_para_check.addLayout(self.layout_h_line_eqpt_para_check_sum)
        self.layout_h_eqpt_para_check.addStretch(1)
        self.layout_h_eqpt_para_check.addLayout(self.layout_h_label_photo_alarm_eqpt_para_check)
        self.layout_h_eqpt_para_check.addStretch(3)

        self.message_title = QLabel()
        self.message_title.setText("<b>客户端实时反馈信息看板<b>")
        self.message_title.setFont(QFont("SanSerif", 10))
        self.message_title.setStyleSheet("Color: RGB(64, 224, 208)")
        self.message_title.setAlignment(Qt.AlignCenter)
        self.message_title.setFixedSize(200, 30)
        self.layout_h_message_title = QHBoxLayout()
        self.layout_h_message_title.addWidget(self.message_title)

        self.text_message = QTextEdit()
        self.text_message.setAlignment(Qt.AlignLeft)
        self.text_message.setPlaceholderText("客户端实时反馈信息待更新......")
        self.text_message.setFont(QFont("微软雅黑", 8))
        self.text_message.setFixedSize(1200, 200)
        self.text_message.setEnabled(False)
        self.layout_h_text_message = QHBoxLayout()
        self.layout_h_text_message.addWidget(self.text_message)

        self.layout_v_message_show = QVBoxLayout()
        self.layout_v_message_show.addLayout(self.layout_h_message_title)
        self.layout_v_message_show.addLayout(self.layout_h_text_message)

        self.label_link_paoliaolv = QLabel()
        self.label_link_paoliaolv.setText(
            '1. AOI 抛料率 By Daily MEDA报表link：<a href="http://10.5.13.172:8086/MEDAPortal/eda_opener.html?now=1644989765570&site=S06&rpt_id=S06_yFD_AOI_PaoliaoLv_Daily">点击打开查看</a>')
        self.label_link_paoliaolv.setGeometry(20, 30, 100, 25)
        self.label_link_paoliaolv.setFixedSize(300, 12)
        self.label_link_paoliaolv.setOpenExternalLinks(True)  # 使其成为超链接
        self.label_link_paoliaolv.linkHovered.connect(self.change_step1)
        self.label_link_paoliaolv.setTextInteractionFlags(Qt.TextBrowserInteraction)  # 双击可选中文本
        self.layout_h_label_link_paoliaolv = QHBoxLayout()
        self.layout_h_label_link_paoliaolv.addWidget(self.label_link_paoliaolv)

        self.button_download_image = QPushButton()
        self.button_download_image.setText("下载不良图片")
        self.button_download_image.clicked.connect(self.download_image)
        self.button_download_image.setToolTip("根据SN自动云端下载不良照片")
        self.button_download_image.setFont(QFont("微软雅黑", 10))
        self.button_download_image.setFixedSize(100, 25)
        self.layout_button_download_image = QHBoxLayout()
        self.layout_button_download_image.addWidget(self.button_download_image)

        self.label_link_longjuanfeng = QLabel()
        self.label_link_longjuanfeng.setText(
            '2. AOI 龙卷风 By Event 通报平台link：<a href="http://10.5.13.48/tornado-cloud/tornado-admin/console.html#policy">点击打开查看</a>')
        self.label_link_longjuanfeng.setGeometry(20, 30, 100, 25)
        self.label_link_longjuanfeng.setFixedSize(400, 12)
        self.label_link_longjuanfeng.setOpenExternalLinks(True)  # 使其成为超链接
        self.label_link_longjuanfeng.linkHovered.connect(self.change_step3)
        self.label_link_longjuanfeng.setTextInteractionFlags(Qt.TextBrowserInteraction)  # 双击可选中文本
        self.layout_h_label_link_longjuanfeng = QHBoxLayout()
        self.layout_h_label_link_longjuanfeng.addWidget(self.label_link_longjuanfeng)

        self.label_link_paoliaolvbyhour = QLabel()
        self.label_link_paoliaolvbyhour.setText(
            '3. AOI 抛料率 By Hours MEDA报表link：<a href="http://10.5.13.172:8086/MEDAPortal/eda_opener.html?now=1644989765570&site=S06&rpt_id=S06_AOI_PaoliaoLv_Daily_By_Hour">点击打开查看</a>')
        self.label_link_paoliaolvbyhour.setGeometry(20, 30, 100, 25)
        self.label_link_paoliaolvbyhour.setFixedSize(400, 12)
        self.label_link_paoliaolvbyhour.setOpenExternalLinks(True)  # 使其成为超链接
        self.label_link_paoliaolvbyhour.linkHovered.connect(self.change_step5)
        self.label_link_paoliaolvbyhour.setTextInteractionFlags(Qt.TextBrowserInteraction)  # 双击可选中文本
        self.layout_h_label_link_paoliaolvbyhour = QHBoxLayout()
        self.layout_h_label_link_paoliaolvbyhour.addWidget(self.label_link_paoliaolvbyhour)

        self.layout_h_link = QHBoxLayout()
        self.layout_h_link.addStretch(8)
        self.layout_h_link.addLayout(self.layout_h_label_link_paoliaolv)
        # self.layout_h_link.addStretch(2)
        self.layout_h_link.addLayout(self.layout_button_download_image)
        self.layout_h_link.addStretch(2)
        self.layout_h_link.addLayout(self.layout_h_label_link_longjuanfeng)
        self.layout_h_link.addStretch(2)
        self.layout_h_link.addLayout(self.layout_h_label_link_paoliaolvbyhour)
        self.layout_h_link.addStretch(1)

        self.button_step_1 = QPushButton()
        self.button_step_1.setText("Step1\n龙卷风通报")
        self.button_step_1.clicked.connect(self.change_step1)
        self.button_step_1.setFont(QFont("微软雅黑", 10))
        self.button_step_1.setFixedSize(150, 50)
        self.layout_button_step_1 = QHBoxLayout()
        self.layout_button_step_1.addWidget(self.button_step_1)

        self.label_step1 = QLabel()
        self.label_step1.setText("<b>----->>><b>")
        self.label_step1.setFont(QFont("SanSerif", 10))
        self.label_step1.setStyleSheet("Color: RGB(64, 224, 208)")
        self.label_step1.setAlignment(Qt.AlignCenter)
        self.label_step1.setFixedSize(70, 50)
        self.layout_h_label_step1 = QHBoxLayout()
        self.layout_h_label_step1.addWidget(self.label_step1)

        self.button_step_2 = QPushButton()
        self.button_step_2.setText("Step2\n远程图像和参数AI分析")
        self.button_step_2.clicked.connect(self.change_step2)
        self.button_step_2.setFont(QFont("微软雅黑", 10))
        self.button_step_2.setFixedSize(200, 50)
        self.layout_button_step_2 = QHBoxLayout()
        self.layout_button_step_2.addWidget(self.button_step_2)

        self.label_step2 = QLabel()
        self.label_step2.setText("<b>----->>><b>")
        self.label_step2.setFont(QFont("SanSerif", 10))
        self.label_step2.setStyleSheet("Color: RGB(64, 224, 208)")
        self.label_step2.setAlignment(Qt.AlignCenter)
        self.label_step2.setFixedSize(70, 50)
        self.layout_h_label_step2 = QHBoxLayout()
        self.layout_h_label_step2.addWidget(self.label_step2)

        self.button_step_3 = QPushButton()
        self.button_step_3.setText("Step3\n任务下发")
        self.button_step_3.clicked.connect(self.change_step3)
        self.button_step_3.setFont(QFont("微软雅黑", 10))
        self.button_step_3.setFixedSize(150, 50)
        self.layout_button_step_3 = QHBoxLayout()
        self.layout_button_step_3.addWidget(self.button_step_3)

        self.label_step3 = QLabel()
        self.label_step3.setText("<b>----->>><b>")
        self.label_step3.setFont(QFont("SanSerif", 10))
        self.label_step3.setStyleSheet("Color: RGB(64, 224, 208)")
        self.label_step3.setAlignment(Qt.AlignCenter)
        self.label_step3.setFixedSize(70, 50)
        self.layout_h_label_step3 = QHBoxLayout()
        self.layout_h_label_step3.addWidget(self.label_step3)

        self.button_step_4 = QPushButton()
        self.button_step_4.setText("Step4\n现场执行和进度回报")
        self.button_step_4.clicked.connect(self.change_step4)
        self.button_step_4.setFont(QFont("微软雅黑", 10))
        self.button_step_4.setFixedSize(200, 50)
        self.layout_button_step_4 = QHBoxLayout()
        self.layout_button_step_4.addWidget(self.button_step_4)

        self.label_step4 = QLabel()
        self.label_step4.setText("<b>----->>><b>")
        self.label_step4.setFont(QFont("SanSerif", 10))
        self.label_step4.setStyleSheet("Color: RGB(64, 224, 208)")
        self.label_step4.setAlignment(Qt.AlignCenter)
        self.label_step4.setFixedSize(70, 50)
        self.layout_h_label_step4 = QHBoxLayout()
        self.layout_h_label_step4.addWidget(self.label_step4)

        self.button_step_5 = QPushButton()
        self.button_step_5.setText("Step5\n效果确认")
        self.button_step_5.clicked.connect(self.change_step5)
        self.button_step_5.setFont(QFont("微软雅黑", 10))
        self.button_step_5.setFixedSize(150, 50)
        self.layout_button_step_5 = QHBoxLayout()
        self.layout_button_step_5.addWidget(self.button_step_5)

        self.layout_step = QHBoxLayout()
        self.layout_step.addStretch(1)
        self.layout_step.addLayout(self.layout_button_step_1)
        self.layout_step.addLayout(self.layout_h_label_step1)
        self.layout_step.addLayout(self.layout_button_step_2)
        self.layout_step.addLayout(self.layout_h_label_step2)
        self.layout_step.addLayout(self.layout_button_step_3)
        self.layout_step.addLayout(self.layout_h_label_step3)
        self.layout_step.addLayout(self.layout_button_step_4)
        self.layout_step.addLayout(self.layout_h_label_step4)
        self.layout_step.addLayout(self.layout_button_step_5)
        self.layout_step.addStretch(1)

    def set_layout(self):
        self.layout_h_title = QHBoxLayout()
        self.layout_h_title.addLayout(self.layout_h_label_title)

        self.layout_h_button_start = QHBoxLayout()
        self.layout_h_button_start.addLayout(self.layout_button_start)

        self.layout_v_windows = QVBoxLayout()
        self.layout_v_windows.addLayout(self.layout_h_title)
        self.layout_v_windows.addStretch(1)
        self.layout_v_windows.addLayout(self.layout_h_button_start)
        self.layout_v_windows.addStretch(1)
        self.layout_v_windows.addLayout(self.layout_h_source_panel_check)
        self.layout_v_windows.addStretch(1)
        self.layout_v_windows.addLayout(self.layout_h_clean_panel_check)
        self.layout_v_windows.addStretch(1)
        self.layout_v_windows.addLayout(self.layout_h_eqpt_para_check)
        self.layout_v_windows.addStretch(1)
        self.layout_v_windows.addLayout(self.layout_v_message_show)
        self.layout_v_windows.addStretch(1)
        self.layout_v_windows.addLayout(self.layout_h_link)
        self.layout_v_windows.addStretch(1)
        self.layout_v_windows.addLayout(self.layout_step)
        self.layout_v_windows.addStretch(1)

    def download_image(self):
        try:
            self.change_step2()
            sn_file_path = QFileDialog.getOpenFileNames(self, "选择不良SN文件所在文件夹")
            sn_file_path = sn_file_path[0][0]
            reader = csv.DictReader(open(sn_file_path, encoding='UTF-8'))

            ftp = FTP()
            ftp.connect(host='10.5.14.55')
            ftp.login("S06FDAOIIMAGE", "*S06FDAOIIMAGE#")
            ftp.cwd("S06/S063BFI01/%s/" % datetime.date.today().strftime('%Y%m%d'))

            for raw in reader:
                sn = raw['LOT_NO']
                print(sn)
                L = ftp.nlst()
                for index, item in enumerate(L):
                    if sn == item.split("_")[0]:
                        print(item)
                        if not os.path.exists(os.getcwd() + "\\image\\"):
                            os.makedirs(str(os.getcwd() + "\\image\\").encode('UTF-8'))
                        ftp.retrbinary("RETR %s" % item, open(os.getcwd() + "\\image\\" + item, "wb").write)

            os.startfile(str(os.getcwd() + "\\image\\"))
        except:
            QMessageBox.about(self, '错误', '不良图片下载失败！')

    def time_start(self):
        self.time_sec = 8 * 60
        now_time = datetime.datetime.strptime('%s:%s' % (self.time_sec // 60, self.time_sec % 60), '%M:%S')

        self.change_step4()

        self.lcd_time = QLCDNumber()
        self.lcd_time.setDigitCount(5)
        self.lcd_time.setMode(QLCDNumber.Dec)
        self.lcd_time.setSegmentStyle(QLCDNumber.Flat)
        self.lcd_time.setFixedSize(300, 100)
        self.lcd_time.display(str(now_time))

        self.layout_button_start.removeWidget(self.button_start)
        sip.delete(self.button_start)
        self.layout_button_start.addWidget(self.lcd_time)

        # 在类中定义一个定时器,并在构造函数中设置启动及其信号和槽
        self.message = ""
        self.name = ""
        self.update_msg_timer = QTimer(self)
        self.update_msg_timer.start(100)
        self.update_msg_timer.timeout.connect(self.update_msg_timeout_slot)

        # 在类中定义一个定时器,并在构造函数中设置启动及其信号和槽
        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout_slot)

    def timeout_slot(self):
        if self.time_sec > 0:
            self.time_sec = self.time_sec - 1
            now_time = datetime.datetime.strptime('%s:%s' % (self.time_sec // 60, self.time_sec % 60), '%M:%S')
            self.lcd_time.display(str(now_time))
        else:
            QMessageBox.about(self, '警告', '本次异常处理超时，请尽快解决！')
            self.timer.stop()

    def update_msg_timeout_slot(self):
        global msg_from_client

        def is_set():
            try:
                type(msg_from_client)
                return 1
            except Exception as e:
                return 0
                pass

        if is_set():
            client = msg_from_client['client']
            port = msg_from_client['port']
            msg = eval(str(str(msg_from_client['msg'])[2:])[:-1].split("[")[1].split("]")[0])
            name = msg['name']
            schedule = msg['schedule']
            total_num = msg['total_num']
            ok_num = msg['ok_num']
            ng_num = msg['ng_num']

            # print(client, port, name, schedule, total_num, ok_num, ng_num)
            if name == 'source_panel_check':  # 来料洁净确认
                if name != self.name:
                    self.message = self.message + "\n" + "来自客户端【%s：%s】的消息：%s" % (client, port, msg)
                    self.text_message.setPlaceholderText(self.message)
                    self.line_source_panel_check_sum.setPlaceholderText("来料洁净抽检完成；抽检数据：%s，OK 数量：%s，NG 数量：%s；状态：%s" % (total_num, ok_num, ng_num, schedule))
                    self.line_source_panel_check_sum.setStyleSheet("background:green")
                    self.label_photo_alarm_source_panel_check.setPixmap(QPixmap("./src/green.png").scaled(80, 80))
                    self.name = name
                    if len(self.message) >= 500:
                        self.timer.stop()
                        self.text_message.setStyleSheet("background:green")
                        QMessageBox.about(self, '提醒', '本次异常处理完成，感谢相关人员的积极配合！')
            if name == 'clean_panel_study':  # 入料清洁手法宣导
                if name != self.name:
                    self.message = self.message + "\n" + "来自客户端【%s：%s】的消息：%s" % (client, port, msg)
                    self.text_message.setPlaceholderText(self.message)
                    self.line_clean_panel_study_sum.setPlaceholderText("入料清洁手法Check完成；状态：%s" % schedule)
                    self.line_clean_panel_study_sum.setStyleSheet("background:green")
                    self.name = name
                    if len(self.message) >= 500:
                        self.timer.stop()
                        self.text_message.setStyleSheet("background:green")
                        QMessageBox.about(self, '提醒', '本次异常处理完成，感谢相关人员的积极配合！')
            if name == 'clean_panel_check':  # 入料清洁确认
                if name != self.name:
                    self.message = self.message + "\n" + "来自客户端【%s：%s】的消息：%s" % (client, port, msg)
                    self.text_message.setPlaceholderText(self.message)
                    self.line_clean_panel_check_sum.setPlaceholderText("入料清洁抽检完成；抽检数据：%s，OK 数量：%s，NG 数量：%s；状态：%s" % (total_num, ok_num, ng_num, schedule))
                    self.line_clean_panel_check_sum.setStyleSheet("background:green")
                    self.label_photo_alarm_clean_panel_check.setPixmap(QPixmap("./src/green.png").scaled(80, 80))
                    self.name = name
                    if len(self.message) >= 500:
                        self.timer.stop()
                        self.text_message.setStyleSheet("background:green")
                        QMessageBox.about(self, '提醒', '本次异常处理完成，感谢相关人员的积极配合！')
            if name == 'eqpt_para_study':  # 机台参数优化
                if name != self.name:
                    self.message = self.message + "\n" + "来自客户端【%s：%s】的消息：%s" % (client, port, msg)
                    self.text_message.setPlaceholderText(self.message)
                    self.line_eqpt_para_study_sum.setPlaceholderText("机台参数优化完成；状态：%s" % schedule)
                    self.line_eqpt_para_study_sum.setStyleSheet("background:green")
                    self.name = name
                    if len(self.message) >= 500:
                        self.timer.stop()
                        self.text_message.setStyleSheet("background:green")
                        QMessageBox.about(self, '提醒', '本次异常处理完成，感谢相关人员的积极配合！')
            if name == 'eqpt_para_check':  # 机台试跑确认
                if name != self.name:
                    self.message = self.message + "\n" + "来自客户端【%s：%s】的消息：%s" % (client, port, msg)
                    self.text_message.setPlaceholderText(self.message)
                    self.line_eqpt_para_check_sum.setPlaceholderText("机台试跑完成；试跑数据：%s，OK 数量：%s，NG 数量：%s；状态：%s" % (total_num, ok_num, ng_num, schedule))
                    self.line_eqpt_para_check_sum.setStyleSheet("background:green")
                    self.label_photo_alarm_eqpt_para_check.setPixmap(QPixmap("./src/green.png").scaled(80, 80))
                    self.name = name
                    if len(self.message) >= 500:
                        self.timer.stop()
                        self.text_message.setStyleSheet("background:green")
                        QMessageBox.about(self, '提醒', '本次异常处理完成，感谢相关人员的积极配合！')

    def change_step1(self):
        self.button_step_1.setStyleSheet("QPushButton{background-color:rgb(0,255,0)}")
        self.button_step_2.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_3.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_4.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_5.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")

    def change_step2(self):
        self.button_step_1.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_2.setStyleSheet("QPushButton{background-color:rgb(0,255,0)}")
        self.button_step_3.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_4.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_5.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")

    def change_step3(self):
        self.button_step_1.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_2.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_3.setStyleSheet("QPushButton{background-color:rgb(0,255,0)}")
        self.button_step_4.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_5.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")

    def change_step4(self):
        self.button_step_1.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_2.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_3.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_4.setStyleSheet("QPushButton{background-color:rgb(0,255,0)}")
        self.button_step_5.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")

    def change_step5(self):
        self.button_step_1.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_2.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_3.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_4.setStyleSheet("QPushButton{background-color:rgb(47,79,79)}")
        self.button_step_5.setStyleSheet("QPushButton{background-color:rgb(0,255,0)}")


if __name__ == '__main__':
    app_system = QApplication(sys.argv)
    form_system = MainWindow()
    form_system.show()
    sys.exit(app_system.exec_())

    pass

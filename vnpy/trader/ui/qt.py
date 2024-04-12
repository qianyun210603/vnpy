import ctypes
import platform
import sys
import traceback
import webbrowser
from types import TracebackType
from typing import Type
import threading

import qdarkstyle
from PySide6 import QtGui, QtWidgets, QtCore

from ..setting import SETTINGS
from ..utility import get_icon_path
from ..locale import _


Qt = QtCore.Qt
QtCore.pyqtSignal = QtCore.Signal
QtWidgets.QAction = QtGui.QAction
QtCore.QDate.toPyDate = QtCore.QDate.toPython
QtCore.QDateTime.toPyDate = QtCore.QDateTime.toPython


def create_qapp(app_name: str = "VeighNa Trader") -> QtWidgets.QApplication:
    """
    Create Qt Application.
    """
    # Set up dark stylesheet
    qapp: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
    qapp.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))

    # Set up font
    font: QtGui.QFont = QtGui.QFont(SETTINGS["font.family"], SETTINGS["font.size"])
    qapp.setFont(font)

    # Set up icon
    icon: QtGui.QIcon = QtGui.QIcon(get_icon_path(__file__, "vnpy.ico"))
    qapp.setWindowIcon(icon)

    # Set up windows process ID
    if "Windows" in platform.uname():
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_name)

    # Hide help button for all dialogs
    # qapp.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton)

    # Exception Handling
    exception_widget: ExceptionWidget = ExceptionWidget()

    def excepthook(exctype: Type[BaseException], value: Exception, tb: TracebackType) -> None:
        """Show exception detail with QMessageBox."""
        sys.__excepthook__(exctype, value, tb)

        msg: str = "".join(traceback.format_exception(exctype, value, tb))
        exception_widget.signal.emit(msg)

    sys.excepthook = excepthook

    if sys.version_info >= (3, 8):

        def threading_excepthook(args: threading.ExceptHookArgs) -> None:
            """Show exception detail from background threads with QMessageBox."""
            sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

            msg: str = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
            exception_widget.signal.emit(msg)

        threading.excepthook = threading_excepthook

    return qapp


class ExceptionWidget(QtWidgets.QWidget):
    """"""

    signal: QtCore.Signal = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        """"""
        super().__init__(parent)

        self.init_ui()
        self.signal.connect(self.show_exception)

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(_("触发异常"))
        self.setFixedSize(600, 600)

        self.msg_edit: QtWidgets.QTextEdit = QtWidgets.QTextEdit()
        self.msg_edit.setReadOnly(True)

        copy_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("复制"))
        copy_button.clicked.connect(self._copy_text)

        community_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("求助"))
        community_button.clicked.connect(self._open_community)

        close_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("关闭"))
        close_button.clicked.connect(self.close)

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addWidget(copy_button)
        hbox.addWidget(community_button)
        hbox.addWidget(close_button)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.msg_edit)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def show_exception(self, msg: str) -> None:
        """"""
        self.msg_edit.setText(msg)
        self.show()

    def _copy_text(self) -> None:
        """"""
        self.msg_edit.selectAll()
        self.msg_edit.copy()

    @staticmethod
    def _open_community() -> None:
        """"""
        webbrowser.open("https://www.vnpy.com/forum/forum/2-ti-wen-qiu-zhu")


class GatewayAppSelectPanel(QtWidgets.QDialog):
    checkbox_values_signal = QtCore.Signal(dict)

    def __init__(
        self, parent=None, available_gateways=None, available_apps=None, selected_gateways=(), selected_apps=()
    ):
        super().__init__(parent)

        import importlib.util  # pylint: disable=import-outside-toplevel

        available_gateways = [
            gw for gw in available_gateways if importlib.util.find_spec(f"vnpy_{gw.lower()}") is not None
        ]
        available_apps = [app for app in available_apps if importlib.util.find_spec(f"vnpy_{app.lower()}") is not None]

        self.setWindowTitle("选择接口和应用")

        # Create first group box
        self.groupBox1 = QtWidgets.QGroupBox("接口", self)
        layout1 = QtWidgets.QGridLayout()
        self.checkboxes1 = self.create_checkboxes(layout1, names=available_gateways, selected=selected_gateways)
        self.groupBox1.setLayout(layout1)

        # Create second group box
        self.groupBox2 = QtWidgets.QGroupBox("应用", self)
        layout2 = QtWidgets.QGridLayout()
        self.checkboxes2 = self.create_checkboxes(layout2, names=available_apps, selected=selected_apps)
        self.groupBox2.setLayout(layout2)

        self.ok_button = QtWidgets.QPushButton("确定", self)
        self.ok_button.clicked.connect(self.emit_checkbox_values)

        # Add group boxes to the dialog's layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.groupBox1)
        layout.addWidget(self.groupBox2)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

        # Dictionary to store checkbox values
        self.checkbox_values = self.get_checkbox_values()

    def create_checkboxes(self, layout, names, selected=(), num_per_row=6):
        checkboxes = []
        for i, name in enumerate(names):
            checkbox = QtWidgets.QCheckBox(name, self)
            if name in selected:
                checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_checkbox_values)
            layout.addWidget(checkbox, i // num_per_row, i % num_per_row)
            checkboxes.append(checkbox)
        return checkboxes

    def update_checkbox_values(self):
        self.checkbox_values = self.get_checkbox_values()

    def get_checkbox_values(self):
        return {
            "gateways": [checkbox.text() for checkbox in self.checkboxes1 if checkbox.isChecked()],
            "apps": [checkbox.text() for checkbox in self.checkboxes2 if checkbox.isChecked()],
        }

    def emit_checkbox_values(self):
        checkbox_values = self.get_checkbox_values()
        if checkbox_values["gateways"] or checkbox_values["apps"]:
            self.checkbox_values_signal.emit(checkbox_values)
            self.close()
        else:
            QtWidgets.QMessageBox.warning(self, "警告", "请至少选择一个接口和一个应用")

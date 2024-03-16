# -*- coding: utf-8 -*-
# @Time    : 2024/3/1 14:57
# @Author  : YQ Tsui
# @File    : utilities.py
# @Purpose : Utility objects for UIs.

from .qt import QtCore, QtWidgets, QtGui
from typing import Any, Tuple, Callable, Dict
from enum import Enum


class RegisteredQWidgetType(Enum):
    """
    Enum for registered widget type.
    """

    GW_EDITBOX = "str"
    GW_PASSWORDBOX = "password"
    GW_COMBOBOX = "list"
    GW_INTBOX = "int"
    GW_FLOATBOX = "float"


class VNPasswordField(QtWidgets.QFrame):

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        self.password_line = QtWidgets.QLineEdit()
        self.password_line.setEchoMode(QtWidgets.QLineEdit.Password)
        self.displayCheckBox = QtWidgets.QCheckBox("显示")
        self.displayCheckBox.stateChanged.connect(self.displayPassword)
        layout.addWidget(self.password_line)
        layout.addWidget(self.displayCheckBox)

        self.setLayout(layout)

    def displayPassword(self, state):
        if state == QtCore.Qt.Checked.value:
            self.password_line.setEchoMode(QtWidgets.QLineEdit.Normal)
        else:
            self.password_line.setEchoMode(QtWidgets.QLineEdit.Password)

    def text(self):
        return self.password_line.text()

    def setText(self, text: str):
        self.password_line.setText(text)


QWIDGET_TYPE_MAPPING = {
    RegisteredQWidgetType.GW_EDITBOX: QtWidgets.QLineEdit,
    RegisteredQWidgetType.GW_INTBOX: QtWidgets.QLineEdit,
    RegisteredQWidgetType.GW_FLOATBOX: QtWidgets.QLineEdit,
    RegisteredQWidgetType.GW_PASSWORDBOX: VNPasswordField,
    RegisteredQWidgetType.GW_COMBOBOX: QtWidgets.QComboBox,
}


class CellStyler:
    """
    Base cell style.
    """

    def __call__(self, cell: QtWidgets.QTableWidgetItem, value: Any) -> QtWidgets.QTableWidgetItem:
        """
        Apply style to cell.
        """
        cell.setText(str(value))
        cell.setTextAlignment(QtCore.Qt.AlignCenter)
        return cell


class CellStylerNumeric(CellStyler):
    """
    Text cell style.
    """

    def __init__(
        self,
        text_formatter: Callable[[Any], str] = None,
        text_color: Tuple[int, int, int] = None,
        background_color: Tuple[int, int, int] = None,
    ) -> None:
        """"""
        self.text_formatter = text_formatter
        self.text_color = QtGui.QBrush(QtGui.QColor(*text_color)) if text_color else None
        self.background_color = QtGui.QColor(*background_color) if background_color else None

    def __call__(self, cell, value) -> QtWidgets.QTableWidgetItem:
        """
        Apply style to text cell.
        """
        if self.text_formatter:
            cell.setText(self.text_formatter(value))
        else:
            cell.setText(str(value))

        if self.text_color:
            cell.setForeground(self.text_color)

        if self.background_color:
            cell.setBackground(self.background_color)

        return cell


class CellStylerConditional(CellStyler):
    """
    Text cell style with conditional text color.
    """

    def __init__(self, conditioner: Callable[[Any], int], styler_mappings: Dict[int, CellStyler]) -> None:
        """"""
        self.conditioner = conditioner
        self.styler_mappings = styler_mappings

    def __call__(self, cell, value) -> QtWidgets.QTableWidgetItem:
        """
        Apply style to text cell.
        """
        condition = self.conditioner(value)
        styler = self.styler_mappings.get(condition, CellStyler())
        return styler(cell, value)

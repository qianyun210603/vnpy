# -*- coding: utf-8 -*-
# @Time    : 2024/3/1 14:57
# @Author  : YQ Tsui
# @File    : utilities.py
# @Purpose : Utility objects for UIs.

from abc import ABC
from .qt import QtCore, QtWidgets
from typing import Any


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


class CellStylerFloat(CellStyler):
    """
    Text cell style.
    """

    def __init__(self, text_formatter=None, text_styles=None, cell_style=None) -> None:
        """"""
        self.text_formatter = text_formatter
        self.text_styles = text_styles
        self.cell_style = cell_style

    def __call__(self, cell, value) -> QtWidgets.QTableWidgetItem:
        """
        Apply style to text cell.
        """
        if self.text_formatter:
            cell.setText(self.text_formatter(value))

        if self.text_styles:
            for style in self.text_styles:
                cell.setStyleSheet(style)

        if self.cell_style:
            cell.setStyle(self.cell_style)

        return cell

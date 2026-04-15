#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Standalone 1D curve processor entry point
独立一维曲线处理器入口

Launch command / 启动命令:
    python -m BGsub.standalone_1d.main
    # or after install:  BGsub-1d
"""

from __future__ import annotations

import sys


def main() -> None:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
    from BGsub.standalone_1d.curve_window import CurveWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = CurveWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

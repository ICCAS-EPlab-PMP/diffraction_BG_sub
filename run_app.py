#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_app.py — BGsub 2D GUI launcher
run_app.py — BGsub 二维程序启动入口
"""

import os
import sys

# Add parent directory to path so we can import BGsub
# 将父目录加入路径以便导入 BGsub 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BGsub.gui.main_window import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_1d.py — BGsub standalone 1D launcher
run_1d.py — BGsub 独立一维程序启动入口
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BGsub.standalone_1d.main import main

if __name__ == "__main__":
    main()

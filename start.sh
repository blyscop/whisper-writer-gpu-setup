#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"
source venv/bin/activate
export QT_QPA_PLATFORM=xcb
python run.py

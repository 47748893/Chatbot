#!/usr/bin/env python
# coding: utf-8

import subprocess
import sys

def install_packages():
    packages = ['transformers', 'torch', 'pandas', 'scikit-learn']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    install_packages()

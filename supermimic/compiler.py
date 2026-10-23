"""Fail before simulation allocation if the CUDA compilation toolchain is missing."""

import os
import shutil


def check_compiler():
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if not cc or not cxx or not shutil.which(cc) or not shutil.which(cxx):
        raise RuntimeError(
            "torch.compile requires C/C++ compilers. Install build-essential and set CC/CXX."
        )
    os.environ.setdefault("CC", cc)
    os.environ.setdefault("CXX", cxx)

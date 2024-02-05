#!/usr/bin/env python3.9

import sys
import shutil
import os

print("start of processing")
src = os.environ['INPUT_DIR']
dest = os.environ['OUTPUT_DIR']

shutil.copytree(src, dest, dirs_exist_ok=True)
print("end of processing")
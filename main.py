#!/usr/bin/env python3.8

import sys
import shutil

print("start of processing")
src = sys.argv[1]
dest = sys.argv[2]

shutil.copytree(src, dest, dirs_exist_ok=True)
print("end of processing")
#!/usr/bin/env python3.8

import sys
import shutil

print("start of processing")
shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True)
print("end of processing")
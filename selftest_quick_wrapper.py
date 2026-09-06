#!/usr/bin/env python3
import os, sys, subprocess
KIT = r'C:\dev\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit'
p = subprocess.run([sys.executable, 'src/selftest.py', '--quick'],
                   cwd=KIT, capture_output=False)
sys.exit(p.returncode)

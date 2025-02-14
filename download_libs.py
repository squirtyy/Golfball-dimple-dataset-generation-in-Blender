import subprocess
import sys
import os
 
# path to python.exe
python_exe = os.path.join(sys.prefix, 'bin', 'python3.11')
 
# upgrade pip
subprocess.call([python_exe, "-m", "ensurepip"])
subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
 
# install required packages
subprocess.call([python_exe, "-m", "pip", "install", "shapely"])
subprocess.call([python_exe, "-m", "pip", "install", "numpy"])
subprocess.call([python_exe, "-m", "pip", "install", "scipy"])
subprocess.call([python_exe, "-m", "pip", "install", "mathutils"])

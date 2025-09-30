import os
import glob

def get_recent_files(directory, file_ext):
    if "." not in file_ext:
        file_ext = f".{file_ext}"
    pattern = os.path.join(directory, f"*{file_ext}")
    ls_all_files = glob.glob(pattern)
    return sorted(ls_all_files,key=os.path.getmtime, reverse=True)

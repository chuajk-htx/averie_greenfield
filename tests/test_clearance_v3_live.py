from datetime import datetime
import os
import sys
from time import sleep
from typing import List
import unittest
from avgf_frontend.get_recent_files import get_recent_files

class TestClearanceV3Live(unittest.TestCase):
    
    def _create_mock_folder_files(self, foldername, list_len) -> List[str]:
        filepaths = []
        dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), foldername)
        os.makedirs(f"{dirpath}", exist_ok=True)
        for i in range(list_len):
            filename = f"testfile_{i+1}.txt"
            filepath = os.path.join(dirpath,filename)
            with open(f"{filepath}",'w'):
                pass
            filepaths.append(filepath)
            sleep(1)
        
        return dirpath, filepaths
    
    def test_get_recent_files(self):
        folderName = "mock_files"
        dirpath, test_filepaths = self._create_mock_folder_files(foldername=folderName, list_len=5)
        expected = sorted(test_filepaths,key=os.path.getctime, reverse=True)
        actual = get_recent_files(directory=dirpath, file_ext="txt")
        self.assertEqual(expected,actual)
        
        
    
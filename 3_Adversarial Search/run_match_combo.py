import sys 
import run_match
import time

import subprocess

file_results=open('report_output.txt','w')

subprocess.call('python run_match.py -r 1',stdout=file_results)

file_results.close()

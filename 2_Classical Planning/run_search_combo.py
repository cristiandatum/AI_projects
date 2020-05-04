
import sys 
import run_search
import time

import subprocess

file_results=open('report_output.txt','w')

for p in range(1,5,1):
    for h in range (1,12,1):

        run_search_dir='python run_search.py'+ ' -p '+ str(p)+ ' -s '+str(s)

        subprocess.call(run_search_dir,stdout=file_results)

file_results.close()
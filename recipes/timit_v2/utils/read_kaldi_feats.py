#!/usr/bin/python3


import sys
import numpy as np
import argparse
import re
import subprocess
import os

def readScp(scpfile, tgtdir):
    with open(scpfile, 'r') as f:
        for i, line in enumerate(f):
            output = subprocess.check_output\
            ("/export/b07/jyang/kaldi-jyang/kaldi/src/featbin/copy-feats \
            --print-args=false scp:\"awk \'NR=={count}\' {f} |\" ark,t:-".format(
            count=i+1, f=scpfile), shell=True)
            uttStr = output.decode('utf-8')
            uttStr = re.sub('(\[|\])', '', uttStr)
            uttList= uttStr.split('\n')
            uttList = list(filter(None, uttList))
            uttName = re.sub(r'\s+', '', uttList.pop(0))
            fname = os.path.join(tgtdir, uttName + '.npy')
            arr= []
            for nk, k in enumerate(uttList):
                perFrame = list(map(float, k.split()))
                arr.append(perFrame)
            arrNp = np.array(arr)
            np.save(fname, arrNp)

def main():
    parser = argparse.ArgumentParser\
    (description = 'Conver Kaldi MFCC features into npz file')
    parser.add_argument('scpfile', help = 'Input MFCC scp file')
    parser.add_argument('tgtdir', help='Target directory')
    args = parser.parse_args()

    readScp(args.scpfile, args.tgtdir)

if __name__ == '__main__':
    main()


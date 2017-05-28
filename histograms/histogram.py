import argparse
import linecache
import os
import sys

import string

parser = argparse.ArgumentParser(description='Print histograms for the specified file.')
parser.add_argument('filename',type=str,help='The filename without the path')

args = parser.parse_args()
filepath = os.path.join('data', 'squad')
filepath = os.path.join(filepath, args.filename)

dicti = {}

with open(filepath) as filereader:
    for line in filereader:
        a = string.split(line)
        try:
            dicti[len(a)] += 1
        except:
            dicti[len(a)] = 1
    #dicti2 = sorted(dicti,reverse=True)


with open('histograms/'+args.filename+'.histogram','w') as hist:
    for i in dicti:
        hist.write('%d: %d\n' % (i, dicti[i]))


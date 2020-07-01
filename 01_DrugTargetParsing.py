# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:09:03 2020

@author: Marc Jerrone Castro
"""
#Importing Packages
import pandas as pd
import numpy as np
import re

'''
Data parser using regex expressions to extract UniprotIDs and their specific type.
Since this research will focus on the connections of these proteins, we have no need to extract the remaining information
The reference for this code was found in https://www.vipinajayakumar.com/parsing-text-with-python/
'''

target = {
    'UIDs': re.compile(r'UNIPROID\t(?P<UniprotID>.*)\n'),
    'TTs': re.compile(r'TARGTYPE\t(?P<TargetType>.*)\n')
    }

def parse(line):
    for key,rx in target.items():
        match = rx.search(line)
        if match:
            return key,match
    return None,None

data = []
with open('input_data/Data Parsing/TTD') as f:
    record = {}
    for line in f:
        key,match = parse(line)
        if key == 'UIDs':
            record['UID'] = match.group('UniprotID')
        elif key == 'TTs':
            record['TT'] = match.group('TargetType')
            data.append(record.copy())
        else:
            continue

df = pd.DataFrame(data)
df = df[1:]
UIDs = df['UID'].values

nonhuman = []
for i in range(len(UIDs)):
    if UIDs[i].find('_HUMAN') == (-1):
        print(UIDs[i])
        nonhuman.append(i)

df = df.drop(nonhuman)
df['UID'] = df['UID'].str.replace('_HUMAN', '')

UIDs = df['UID'].values

conversion = []
for i in range(len(UIDs)):
    dummy = str(UIDs[i])
    converted = re.sub('\W+',',', dummy)
    converted = converted.replace('_',',')
    conversion.append(converted)

df['UID'] = conversion
df = df.assign(UID=df['UID'].str.split(',')).explode('UID')
df['UID'] = df['UID'].astype(str) + '_HUMAN'
#print(df)
#df.to_csv('parse_data/parse.TTD.targets', sep='\t', encoding='unicode_escape', index = False)

with open('parse_data/parse.TTD.targets', 'w') as filehandle:
    filehandle.writelines('UniprotID\tTargetType\n')
    for i in range(len(df)):
        filehandle.writelines("%s\t" % df['UID'].values[i])
        filehandle.writelines("%s\n" % df['TT'].values[i])


with open('parse_data/parse.TTD.id', 'w') as f:
    proteins = df['UID'].values
    for x in proteins:
        f.writelines("{},".format(x))
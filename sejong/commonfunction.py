# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:45:33 2018

@author: 경인
"""

def getSample(fileNum, disease):
    file_name = ""
    if fileNum < 10:
        file_name = 'sample/'+disease+'_00'+str(fileNum)+'.wav'
    elif fileNum < 100:    
        file_name = 'sample/'+disease+'_0'+str(fileNum)+'.wav'
    else:
        file_name = 'sample/'+disease+'_'+str(fileNum)+'.wav'
    return file_name

    
def returnDisease(disease):
    if disease == 0:
        return 'N'
    elif disease == 1:
        return 'AS'
    elif disease == 2:
        return 'MR'
    elif disease == 3:
        return 'MS'
    elif disease == 4:
        return 'MVP'


def returnLabel(disease) :
    if disease == 0:
        return [1, 0, 0, 0, 0]
    elif disease == 1:
        return [0, 1, 0, 0, 0]
    elif disease == 2:
        return [0, 0, 1, 0, 0]
    elif disease == 3:
        return [0, 0, 0, 1, 0]
    elif disease == 4:
        return [0, 0, 0, 0, 1]
import numpy as np
import librosa
from commonfunction import *

wavlength = 2.5
disease_cnt = 180


for loop in range(5):   
    disease = ''
    array_test = []
    array_test_l = []
    array_all = []
    array_all_l = []
    label = []
    
    for diseaseNum in range(5):
        label = returnLabel(diseaseNum)
        disease = returnDisease(diseaseNum)
        test_size = disease_cnt*0.2
        for filecnt in range(1, disease_cnt+1):        
            file_name = getSample(filecnt, disease)
            
            X, sample_rate = librosa.load(file_name, sr=None)
            
            duration = librosa.get_duration(y=X, sr=sample_rate)
            duration_rate = 1/(wavlength/duration)
            X_2 = librosa.effects.time_stretch(X, duration_rate)        
            
            mfcc = librosa.feature.mfcc(y=X_2, sr=sample_rate, n_mfcc=13, n_mels=26, n_fft=800, hop_length=800)
            mfcc = np.resize(mfcc, (13, 25))
            mfcc = np.resize(mfcc, (325,))
            mfcc_list = mfcc.tolist()
            
            if filecnt>loop*test_size and filecnt <= (loop+1)*test_size:  
                array_test.append(mfcc_list)    
                array_test_l.append(label)
            else:
                array_all.append(mfcc_list)
                array_all_l.append(label)
    
    nparray = np.array(array_all)
    nparray_l = np.array(array_all_l)
    nparray_test = np.asarray(array_test)
    nparray_test_l = np.asarray(array_test_l)
    
    np.savez('data/forcertify_'+str(loop)+'_train', X=nparray, Y=nparray_l)
    np.savez('data/forcertify_'+str(loop)+'_test', X=nparray_test, Y=nparray_test_l)
    
    print('Fold #'+str(loop)+' is done!')
    
    del array_all
    del array_all_l
    del array_test
    del array_test_l
    del nparray
    del nparray_l
    del nparray_test
    del nparray_test_l
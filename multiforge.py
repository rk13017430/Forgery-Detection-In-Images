import sys
import math
import numpy as np
import numpy
import cv2
from PIL import Image
from scipy import signal
from sklearn.cluster import KMeans
from flask import Flask,request
from io import BytesIO
import requests
import pandas as pd
import argparse
import csv
from scipy import fftpack as fftp
from matplotlib import pyplot as plt
app = Flask(__name__)
def detect_compression(input):
    firstq = 30
    secondq = 40
    thres = 0.5

    dct_rows = 0;
    dct_cols = 0;
    image =  cv2.imdecode(np.frombuffer(input, np.uint8), -1)
    # image = cv2.imread(BytesIO(input))
    shape = image.shape;

    if shape[0]%8 != 0: dct_rows = shape[0]+8-shape[0]%8
    else: dct_rows = shape[0]

    if shape[1]%8 != 0: dct_cols = shape[1]+8-shape[1]%8
    else: dct_cols = shape[1]

    dct_image = np.zeros((dct_rows,dct_cols,3),np.uint8)
    dct_image[0:shape[0], 0:shape[1]] = image

    y = cv2.cvtColor(dct_image, cv2.COLOR_BGR2YCR_CB)[:,:,0]

    w = y.shape[1]
    h = y.shape[0]
    n = w*h/64

    Y = y.reshape(h//8,8,-1,8).swapaxes(1,2).reshape(-1, 8, 8)

    qDCT =[]

    for i in range(0,Y.shape[0]): qDCT.append(cv2.dct(np.float32(Y[i])))

    qDCT = np.asarray(qDCT, dtype=np.float32)
    qDCT = np.rint(qDCT - np.mean(qDCT, axis = 0)).astype(np.int32)
    f,a1 = plt.subplots(8,8)
    a1 = a1.ravel()
    k=0;
    flag = True
    for idx,ax in enumerate(a1):
        k+=1;
        data = qDCT[:,int(idx/8),int(idx%8)]
        val,key = np.histogram(data, bins=np.arange(data.min(), data.max()+1),normed = True)
        z = np.absolute(fftp.fft(val))
        z = np.reshape(z,(len(z),1))
        rotz = np.roll(z,int(len(z)/2))

        slope = rotz[1:] - rotz[:-1]
        indices = [i+1 for i in range(len(slope)-1) if slope[i] > 0 and slope[i+1] < 0]
        peak_count = 0
        for j in indices:
            if rotz[j][0]>thres: peak_count+=1

        if(k==3):
            if peak_count>=20: return True
            else: return False
            flag = False
def estimate_noise(I):
    H, W = I.shape
    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    return sigma
def detect_noise(input, blockSize=32):
    im = Image.open(BytesIO(input))
    im = im.convert('1')
    blocks = []
    imgwidth, imgheight = im.size
    # break up image into NxN blocks, N = blockSize
    for i in range(0,imgheight,blockSize):
        for j in range(0,imgwidth,blockSize):
            box = (j, i, j+blockSize, i+blockSize)
            b = im.crop(box)
            a = np.asarray(b).astype(int)
            blocks.append(a)
    variances = []
    for block in blocks:
        variances.append([estimate_noise(block)])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(variances)
    center1, center2 = kmeans.cluster_centers_
    if abs(center1 - center2) > .4: return True
    else: return False
@app.route('/')
def hello_world():
    return 'Hello, World!'
@app.route('/detect', methods=['POST'])
def detect_f():
    result = []
    data = request.json['img']
    print(data)
    for d in data:
        forge = {}
        response = requests.get(d)
        print(type(response.content))
        res1 = detect_noise(response.content)
        res2 = detect_compression(response.content)
        forge["noise"] = res1
        forge["compression"] = res2
        result.append(forge)
    print(result)
    out = {"out":result}
    return out
app.run(host='0.0.0.0',debug=True)
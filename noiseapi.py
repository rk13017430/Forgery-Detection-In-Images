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
app = Flask(__name__)
def estimate_noise(I):
    H, W = I.shape
    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    return sigma
def detect(input, blockSize=32):
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
    if abs(center1 - center2) > .36: return True
    else: return False
# print(detect("dummy.jpeg"))
@app.route('/')
def hello_world():
    return 'Hello, World!'
@app.route('/detect', methods=['POST'])
def detect_f():
    data = request.files['img'].read()
    npimg = np.fromstring(data, numpy.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    # print(img)
    res = detect(data)
    # print(res)
    return {"forged":res}
app.run(host='0.0.0.0',debug=True)
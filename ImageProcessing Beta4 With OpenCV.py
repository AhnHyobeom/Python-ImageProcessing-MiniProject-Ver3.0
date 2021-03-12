from tkinter import *  # GUI 필수
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *
import cv2, glob, dlib
import numpy as np
import colorsys
from matplotlib import pyplot as plt
import argparse

## 함수 선언부

## 공통 함수부
def malloc(row, col, init = 0) :
    retAry = [[[init for _ in range(col)] for _ in range(row)] for _ in range(RGB)]
    return retAry

def shutDown() :
    sys.exit()

def saveImage() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if fileName == None :
        return
    saveCvImage = np.zeros((outH, outW, 3), np.uint8) # 데이터형식 지원
    for i in range(outH) :
        for j in range(outW) :
            tup = tuple(  (  [outImage[BB][i][j], outImage[GG][i][j], outImage[RR][i][j] ] ) )
            # --> ( ( [100, 77, 55] ) )
            saveCvImage[i,j] = tup
    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='.png',
                filetypes=(("Image Type", "*.png;*.jpg;*.bmp;*.tif"),("All File", "*.*")))
    if saveFp == '' or saveFp == None :
        return
    cv2.imwrite(saveFp.name, saveCvImage)
    print('Save ~~')

def openImage() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    fileName = askopenfilename(parent=window,
                               filetypes=(('Color File', '*.png;*.jpg;*.bmp;*.tif'), ('All File', '*.*')))
    if(fileName == "" or fileName == None) :
        return
    # File --> CV 객체
    cvInImage = cv2.imread(fileName)
    # 입력 영상 크기 알아내기 (중요!)
    inH, inW = cvInImage.shape[:2]
    # 메모리 확보
    inImage = malloc(inH, inW)
    # OpenCV --> 메모리
    for i in range(inH):
        for k in range(inW):
            inImage[BB][i][k] = cvInImage.item(i, k, RR)
            inImage[GG][i][k] = cvInImage.item(i, k, GG)
            inImage[RR][i][k] = cvInImage.item(i, k, BB)

    # print(inImage[RR][100][100], inImage[GG][100][100], inImage[BB][100][100])
    equal_image()

def displayImage() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    ## 기존에 그림을 붙인적이 있으면, 게시판(canvas) 뜯어내기
    if canvas != None :
        canvas.destroy()
    window.geometry(str(outW) + "x" + str(outH))
    canvas = Canvas(window, height=outH, width=outW)
    paper = PhotoImage(height=outH, width=outW)
    canvas.create_image((outW/2, outH/2), image=paper, state='normal')
    rgbString = "" # 전체 펜을 저장함
    for i in range(outH) :
        tmpString = "" # 각 1줄의 펜
        for j in range(outW) :
            rr = int(outImage[RR][i][j])
            gg = int(outImage[GG][i][j])
            bb = int(outImage[BB][i][j])
            tmpString += "#%02x%02x%02x " % (rr, gg, bb) # 제일 뒤 공백 1칸
        rgbString += '{' + tmpString + '} ' # 제일 뒤 공백 1칸
    paper.put(rgbString)
    canvas.pack()
    status.configure(text=str(outW) + 'x' + str(outH) + '  ' + fileName)

## 영상처리 함수 모음

## 화소점 처리
def equal_image() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None :
        return
    # 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH; outW = inW;
    outImage = malloc(outH, outW)
    for rgb in range(RGB) :
        for i in range(inH) :
            for j in range(inW) :
                outImage[rgb][i][j] = inImage[rgb][i][j]

    displayImage()

def grayscale_image() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None :
        return
    # 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH; outW = inW;
    outImage = malloc(outH, outW)
    for i in range(inH):
        for j in range(inW):
            hap = inImage[RR][i][j] + inImage[GG][i][j] + inImage[BB][i][j]
            outImage[RR][i][j] = outImage[GG][i][j] = outImage[BB][i][j] = hap // 3  # 몫만
    displayImage()

def add_image() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    # 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    value = askinteger("밝게/어둡게", "값 입력", minvalue=-255, maxvalue=255)
    for rgb in range(RGB):
        for i in range(inH):
            for j in range(inW):
                if (inImage[rgb][i][j] + value > 255):
                    outImage[rgb][i][j] = 255
                elif (inImage[rgb][i][j] + value < 0):
                    outImage[rgb][i][j] = 0
                else:
                    outImage[rgb][i][j] = inImage[rgb][i][j] + value
    displayImage()

def reverse_image() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG ,BB
    global boxLine
    global sx, sy, ex, ey

    sx, sy, ex, ey = [-1] * 4
    boxLine = None  # 처음인지 체크하기
    if (inImage == None) :
        return
    def reverse_image_click(event):
        global sx, sy, ex, ey
        sx = event.x
        sy = event.y
    def reverse_image_release(event):
        global sx, sy, ex, ey
        ex = event.x
        ey = event.y
        if sx > ex :
            sx, ex = ex, sx
        if sy > ey :
            sy, ey = ey, sy
        __reverse_image()
        canvas.unbind("<Button-1>")
        canvas.unbind("<Button-3>")
        canvas.unbind("<ButtonRelease-1>")

    def reverse_image_Rclick(event):
        global sx, sy, ex, ey
        global inH, inW
        sx = 0
        sy = 0
        ex = inW -1
        ey = inH -1
        __reverse_image()
        canvas.unbind("<Button-1>")
        canvas.unbind("<Button-3>")
        canvas.unbind("<ButtonRelease-1>")

    def __reverse_image():
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
        global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
        global sx, sy, ex, ey
        if (inImage == None):
            return
        outH = inH;
        outW = inW;
        outImage = malloc(outH, outW)
        for rgb in range(RGB):
            for i in range(inH):
                for j in range(inW):
                    if ( sx <= j <= ex ) and ( sy <= i <= ey ) :
                        outImage[rgb][i][j] = 255-inImage[rgb][i][j]
                    else :
                        outImage[rgb][i][j] = inImage[rgb][i][j]
        displayImage()

    def mouseMove(event):
        global sx, sy, ex, ey
        global boxLine
        if sx < 0 :
            return
        ex = event.x
        ey = event.y
        if not boxLine : # 기존 박스선 지우기
            pass
        else :
            canvas.delete(boxLine)
        boxLine = canvas.create_rectangle(sx,sy,ex,ey, fill=None)
    canvas.bind("<Button-1>", reverse_image_click)
    canvas.bind("<B1-Motion>", mouseMove)
    canvas.bind("<Button-3>", reverse_image_Rclick)
    canvas.bind("<ButtonRelease-1>", reverse_image_release)

def bw_image() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    # 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    for rgb in range(RGB):
        for i in range(inH):
            for j in range(inW):
                if inImage[rgb][i][j] > 127 :
                    outImage[rgb][i][j] = 255
                else :
                    outImage[rgb][i][j] = 0

    displayImage()

def change_satur() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    # 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    value = askfloat("회전", "값 입력", minvalue=-2.0, maxvalue=2.0)
    for i in range(inH):
        for j in range(inW):
            red = inImage[RR][i][j] / 255
            green = inImage[GG][i][j] / 255
            blue = inImage[BB][i][j] / 255
            h, s, v = colorsys.rgb_to_hsv(red, green, blue)
            s += value
            if s > 1.0 :
                s = 1.0
            if s < 0.0 :
                s = 0.0
            red, green, blue = colorsys.hsv_to_rgb(h, s, v)
            outImage[RR][i][j] = red * 255
            outImage[GG][i][j] = green * 255
            outImage[BB][i][j] = blue * 255
    displayImage()

## 기하학 처리
def sizeUp() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    # 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH * 2
    outW = inW * 2
    outImage = malloc(outH, outW)
    for rgb in range(RGB):
        for i in range(outH):
            for j in range(outW):
                outImage[rgb][i][j] = inImage[rgb][i // 2][j // 2]

    displayImage()

def sizeDown() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    # 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH // 2
    outW = inW // 2
    outImage = malloc(outH, outW)
    for rgb in range(RGB):
        for i in range(outH):
            for j in range(outW):
                outImage[rgb][i][j] = inImage[rgb][i * 2][j * 2]

    displayImage()

def rotate():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    degree = askfloat("회전", "값 입력", minvalue=-360.0, maxvalue=360.0)
    center_w = inW // 2
    center_h = inH // 2
    pi = 3.14159
    seta = pi / (180.0 / degree)
    for i in range(inH):
        for j in range(inW):
            new_w = int((i - center_h) * np.sin(seta) + (j - center_w) * np.cos(seta) + center_w)
            new_h = int((i - center_h) * np.cos(seta) - (j - center_w) * np.sin(seta) + center_h)
            if new_w < 0 :
                continue
            if new_w >= inW :
                continue
            if new_h < 0 :
                continue
            if new_h >= inH :
                continue
            outImage[RR][i][j] = inImage[RR][new_h][new_w]
            outImage[GG][i][j] = inImage[GG][new_h][new_w]
            outImage[BB][i][j] = inImage[BB][new_h][new_w]

    displayImage()

## 영역 처리
def maskOP(mask):
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    sumR, sumG, sumB = [0] * 3
    for i in range(2, inH - 2, 1) :
        for j in range(2, inW - 2, 1) :
            for k in range(5) :
                for m in range(5) :
                    sumR += inImage[RR][i - 2 + k][j - 2 + m] * mask[k][m]
                    sumG += inImage[GG][i - 2 + k][j - 2 + m] * mask[k][m]
                    sumB += inImage[BB][i - 2 + k][j - 2 + m] * mask[k][m]
            outImage[RR][i][j] = sumR
            outImage[GG][i][j] = sumG
            outImage[BB][i][j] = sumB
            sumR, sumG, sumB = [0] * 3

def overflowCheck():
    for rgb in range(RGB):
        for i in range(outH):
            for j in range(outW):
                if outImage[rgb][i][j] > 255 :
                    outImage[rgb][i][j] = 255
                elif outImage[rgb][i][j] < 0 :
                    outImage[rgb][i][j] = 0

def emboss() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    mask = [[-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]]
    maskOP(mask)
    overflowCheck()
    displayImage()

def blur() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    mask = [[1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0],
            [1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0],
            [1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0],
            [1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0],
            [1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0]]
    maskOP(mask)
    overflowCheck()
    displayImage()

def sharp() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    mask = [[-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 24, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]]
    maskOP(mask)
    overflowCheck()
    displayImage()

## 모폴로지
def erosion() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    for rgb in range(3) :
        for i in range(2, inH - 2, 1):
            for j in range(2, inW - 2, 1):
                min = 255;
                for k in range(5):
                    for m in range(5):
                        if inImage[rgb][i - 2 + k][j - 2 + m] < min :
                            min = inImage[rgb][i - 2 + k][j - 2 + m]
                outImage[rgb][i][j] = min
    overflowCheck()
    displayImage()

def dilation() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    for rgb in range(3):
        for i in range(2, inH - 2, 1):
            for j in range(2, inW - 2, 1):
                max = 0;
                for k in range(5):
                    for m in range(5):
                        if inImage[rgb][i - 2 + k][j - 2 + m] > max:
                            max = inImage[rgb][i - 2 + k][j - 2 + m]
                outImage[rgb][i][j] = max
    overflowCheck()
    displayImage()

# 히스토그램
def histoGraph() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if outImage == None:
        return
    rHist = [0 for _ in range(256)]
    gHist = [0 for _ in range(256)]
    bHist = [0 for _ in range(256)]
    for i in range(inH):
        for j in range(inW):
            rHist[int(outImage[RR][i][j])] += 1
            gHist[int(outImage[GG][i][j])] += 1
            bHist[int(outImage[BB][i][j])] += 1
    # 차트 종류, 제목, 차트 크기, 범례, 폰트 크기 설정
    HistoX = [0 for _ in range(len(rHist))]
    for i in range(len(rHist)):
        HistoX[i] = i
    plt.xlabel('Brightness Value')
    plt.ylabel('Brightness sum')
    plt.title('Histogram')
    plt.plot(HistoX, rHist, color='red')
    plt.plot(HistoX, gHist, color='green')
    plt.plot(HistoX, bHist, color='blue')
    plt.legend(['Red', 'Green', 'Blue'])
    plt.show()

def stretching() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    rMax, gMax, bMax = [0] * 3
    rMin, gMin, bMin = [255] * 3
    for i in range(inH):
        for j in range(inW):
            if inImage[RR][i][j] > rMax :
                rMax = inImage[RR][i][j]
            if inImage[GG][i][j] > gMax :
                gMax = inImage[GG][i][j]
            if inImage[BB][i][j] > bMax :
                bMax = inImage[BB][i][j]
            if inImage[RR][i][j] < rMin :
                rMin = inImage[RR][i][j]
            if inImage[GG][i][j] < gMin :
                gMin = inImage[GG][i][j]
            if inImage[BB][i][j] < bMin :
                bMin = inImage[BB][i][j]
    for i in range(inH):
        for j in range(inW):
            outImage[RR][i][j] = ((inImage[RR][i][j] - rMin) / (rMax - rMin)) * 255
            outImage[GG][i][j] = ((inImage[GG][i][j] - gMin) / (gMax - gMin)) * 255
            outImage[BB][i][j] = ((inImage[BB][i][j] - bMin) / (bMax - bMin)) * 255
    displayImage()

def endInSearch() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    rMax, gMax, bMax = [0] * 3
    rMin, gMin, bMin = [255] * 3
    for i in range(inH):
        for j in range(inW):
            if inImage[RR][i][j] > rMax:
                rMax = inImage[RR][i][j]
            if inImage[GG][i][j] > gMax:
                gMax = inImage[GG][i][j]
            if inImage[BB][i][j] > bMax:
                bMax = inImage[BB][i][j]
            if inImage[RR][i][j] < rMin:
                rMin = inImage[RR][i][j]
            if inImage[GG][i][j] < gMin:
                gMin = inImage[GG][i][j]
            if inImage[BB][i][j] < bMin:
                bMin = inImage[BB][i][j]
    rMin -= 50; gMin -= 50; bMin -= 50;
    rMax += 50; gMax += 50; bMax += 50;
    for i in range(inH):
        for j in range(inW):
            outImage[RR][i][j] = ((inImage[RR][i][j] - rMin) / (rMax - rMin)) * 255
            outImage[GG][i][j] = ((inImage[GG][i][j] - gMin) / (gMax - gMin)) * 255
            outImage[BB][i][j] = ((inImage[BB][i][j] - bMin) / (bMax - bMin)) * 255
    overflowCheck()
    displayImage()

def equalized() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    outH = inH
    outW = inW
    outImage = malloc(outH, outW)
    rHist = [0 for _ in range(256)]
    rHistSum = [0 for _ in range(256)]
    gHist = [0 for _ in range(256)]
    gHistSum = [0 for _ in range(256)]
    bHist = [0 for _ in range(256)]
    bHistSum = [0 for _ in range(256)]
    for i in range(inH):
        for j in range(inW):
            rHist[int(inImage[RR][i][j])] += 1
            gHist[int(inImage[GG][i][j])] += 1
            bHist[int(inImage[BB][i][j])] += 1
    rHistSum[0] = rHist[0]
    gHistSum[0] = gHist[0]
    bHistSum[0] = bHist[0]
    for i in range(1, len(rHistSum), 1):
        rHistSum[i] = rHistSum[i - 1] + rHist[i]
        gHistSum[i] = gHistSum[i - 1] + gHist[i]
        bHistSum[i] = bHistSum[i - 1] + bHist[i]
    for i in range(inH):
        for j in range(inW):
            outImage[RR][i][j] = rHistSum[int(inImage[RR][i][j])] / (inH * inW) * 255
            outImage[GG][i][j] = gHistSum[int(inImage[GG][i][j])] / (inH * inW) * 255
            outImage[BB][i][j] = bHistSum[int(inImage[BB][i][j])] / (inH * inW) * 255
    overflowCheck()
    displayImage()

# OpenCV
def cv2outImage() : # OpenCV Out --> 내 메모리 Out
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    # 출력 영상 크기
    outH, outW = cvOutImage.shape[:2]
    # 출력 메모리 할당
    outImage = malloc(outH, outW)
    # CV2 결과 --> 출력 메모리
    for i in range(outH) :
        for j in range(outW) :
            # CV2가 흑백(2차원) 또는 칼라(3차원) 체크
            if(cvOutImage.ndim == 2) :
                outImage[RR][i][j] = cvOutImage.item(i, j)
                outImage[GG][i][j] = cvOutImage.item(i, j)
                outImage[BB][i][j] = cvOutImage.item(i, j)
            else :
                outImage[RR][i][j] = cvOutImage.item(i, j, BB)
                outImage[GG][i][j] = cvOutImage.item(i, j, GG)
                outImage[BB][i][j] = cvOutImage.item(i, j, RR)

# 화소점 처리
def bwCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    gray = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2GRAY)
    ret, cvOutImage = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cv2outImage()
    displayImage()

def adaptiveBwCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    gray = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2GRAY)
    cvOutImage = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 467, 37)
    cv2outImage()
    displayImage()

def equal_imageCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cvInImage.copy()
    cv2outImage()
    displayImage()

def grayscale_imageCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2GRAY)
    cv2outImage()
    displayImage()

def cartoon_imageCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2GRAY)
    cvOutImage = cv2.medianBlur(cvOutImage, 7) # 수치 변경 가능
    edges = cv2.Laplacian(cvOutImage, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    cvOutImage = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2outImage()
    displayImage()



# 기하학 처리
def rotateCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    height, width, channel = cvInImage.shape
    degree = askfloat("회전", "값 입력", minvalue=-360.0, maxvalue=360.0)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    cvOutImage = cv2.warpAffine(cvInImage, matrix, (width, height))
    cv2outImage()
    displayImage()

def sizeUpCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    height, width, channel = cvInImage.shape
    cvOutImage = cv2.pyrUp(cvInImage, dstsize=(width * 2, height * 2), borderType=cv2.BORDER_DEFAULT)
    cv2outImage()
    displayImage()

def sizeDownCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cv2.pyrDown(cvInImage)
    cv2outImage()
    displayImage()

# 영역처리
def emboss_CV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    mask = np.zeros((3,3), np.float32)
    mask[0][0] = -1.0
    mask[2][2] = 1.0
    cvOutImage = cv2.filter2D(cvInImage, -1, mask)
    cvOutImage += 127
    cv2outImage()
    displayImage()

def blur_CV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cv2.blur(cvInImage, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
    cv2outImage()
    displayImage()

def gaussianBlurCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cv2.GaussianBlur(cvInImage, (9, 9), 0)
    cv2outImage()
    displayImage()

def medianBlurCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cv2.medianBlur(cvInImage, 9)
    cv2outImage()
    displayImage()

def bilateralFilterCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cv2.bilateralFilter(cvInImage, 9, 75, 75)
    cv2outImage()
    displayImage()

# 모폴로지
def erosionCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    cvOutImage = cv2.erode(cvInImage, kernel, anchor=(-1, -1), iterations=5)
    cv2outImage()
    displayImage()

def dilationCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    cvOutImage = cv2.dilate(cvInImage, kernel, anchor=(-1, -1), iterations=5)
    cv2outImage()
    displayImage()

# 검출 및 추출
def contoursCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cvInImage.copy()
    gray = cv2.cvtColor(cvOutImage, cv2.COLOR_RGB2GRAY)
    # 이미지에 따라 thresh 조정 colorball.png -> 80, OpenCV_Logo.png -> 180
    ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        cv2.drawContours(cvOutImage, [contours[i]], -1, (0, 255, 255), 4)
    cv2outImage()
    displayImage()

def findCornerCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cvInImage.copy()
    gray = cv2.cvtColor(cvInImage, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)
    for i in corners:
        cv2.circle(cvOutImage, tuple(i[0]), 3, (0, 0, 255), 2)
    cv2outImage()
    displayImage()

def convexHullCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cvInImage.copy()
    gray = cv2.cvtColor(cvOutImage, cv2.COLOR_RGB2GRAY)
    # 이미지에 따라 thresh 조정
    ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        hull = cv2.convexHull(i, clockwise=True)
        cv2.drawContours(cvOutImage, [hull], 0, (0, 255, 255), 4)
    cv2outImage()
    displayImage()

def colorExtractionCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    img_hsv = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2HSV)  # cvtColor 함수를 이용하여 hsv 색공간으로 변환
    # 빨강 노랑 주황 초록 파랑 보라
    colorInput = askstring("색 추출", "두글자로 색 입력(빨강 노랑 주황 초록 파랑 보라)")
    colorIndex = 0;
    if colorInput == "빨강" :
        colorIndex = 170
    elif colorInput == "주황" :
        colorIndex = 15
    elif colorInput == "노랑" :
        colorIndex = 30
    elif colorInput == "초록" :
        colorIndex = 60
    elif colorInput == "파랑" :
        colorIndex = 120
    else : # 보라
        colorIndex = 135
    lower_blue = (colorIndex - 10, 30, 30)  # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_blue = (colorIndex + 10, 255, 255)
    img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)  # 범위내의 픽셀들은 흰색, 나머지 검은색
    # 바이너리 이미지를 마스크로 사용하여 원본이미지에서 범위값에 해당하는 영상부분을 획득
    cvOutImage = cv2.bitwise_and(cvInImage, cvInImage, mask=img_mask)
    cv2outImage()
    displayImage()

def findCircleCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    cvOutImage = cvInImage.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2GRAY)
    gray = cv2.dilate(gray, kernel, anchor=(-1, -1), iterations=3)
    gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_REFLECT101)
    gray = cv2.erode(gray, kernel, anchor=(-1, -1), iterations=3)
    # (검출 이미지, 검출 방법, 해상도 비율, 최소 거리(검출된 원과 원 사이의 최소 거리), 캐니 엣지 임곗값, 중심 임곗값, 최소 반지름, 최대 반지름)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=35, minRadius=0, maxRadius=0)
    for i in circles[0]:
        cv2.circle(cvOutImage, (i[0], i[1]), int(i[2]), (255, 255, 255), 4)
    cv2outImage()
    displayImage()


# 머신러닝 Haarcascade
def face_imageCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    gray = cv2.cvtColor(cvInImage, cv2.COLOR_RGB2GRAY)
    # 얼굴 찾기
    face_rects = face_cascade.detectMultiScale(gray, 1.1, 5)
    cvOutImage = cvInImage.copy()
    for (x,y,w,h) in face_rects :
        cv2.rectangle(cvOutImage, (x, y), (x+w, y+h), (0, 255, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = cvOutImage[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # 원본 이미지에 얼굴의 위치를 표시합니다. ROI에 표시하면 원본 이미지에도 표시됩니다.
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 3)
        roi_grayN = gray[y:y + h, x:x + w]
        roi_colorN = cvOutImage[y:y + h, x:x + w]
        nose = nose_cascade.detectMultiScale(roi_grayN)
        for (nx, ny, nw, nh) in nose:
            # 원본 이미지에 얼굴의 위치를 표시합니다. ROI에 표시하면 원본 이미지에도 표시됩니다.
            cv2.rectangle(roi_colorN, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)
    cv2outImage()
    displayImage()

def catFace_imageCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return
    SF = 1.3
    N = 2
    MS = (220, 220) # 최소 사이즈 이 이하는 전부 무시
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    gray = cv2.cvtColor(cvInImage, cv2.COLOR_RGB2GRAY)
    # 얼굴 찾기
    face_rects = face_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=MS)
    cvOutImage = cvInImage.copy()
    for (x,y,w,h) in face_rects :
        cv2.rectangle(cvOutImage, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2outImage()
    displayImage()

def age_imageCV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if inImage == None:
        return

    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']
    detector = dlib.get_frontal_face_detector()
    age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    faces = detector(cvInImage)
    cvOutImage = cvInImage.copy()

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = cvOutImage[y1:y2, x1:x2].copy()
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
                                     mean=(78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False, crop=False)
        # predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # visualize
        cv2.rectangle(cvOutImage, (x1, y1), (x2, y2), (255, 255, 255), 2)
        overlay_text = '%s %s' % (gender, age)
        cv2.putText(cvOutImage, overlay_text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 0), thickness=4)
        cv2.putText(cvOutImage, overlay_text, org=(x1, y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2outImage()
    displayImage()

def ageVideo_imageCV() :
    def highlightFace(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        return frameOpencvDnn, faceBoxes

    parser = argparse.ArgumentParser()
    parser.add_argument('--image')

    args = parser.parse_args()

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(args.image if args.image else 0)
    padding = 20
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                         :min(faceBox[2] + padding, frame.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)
    return

## 전역 변수부
window, canvas, paper = None, None, None #  벽, 게시판, 종이
inImage, outImage = None, None  # 3차원 배열
inH, inW, outH, outW = [0] * 4
fileName = None
inCvImage, outCvImage = None, None  # openCV 용
RGB, RR, GG ,BB = 3, 0, 1, 2 # 상수

## 메인 코드부
window = Tk(); window.title("영상처리(파이썬) Beta1"); window.geometry('500x500')
window.resizable(width=False, height=False)
status = Label(window, text="이미지정보:", bd=1, relief=SUNKEN, anchor=W)
status.pack(side=BOTTOM, fill=X)

mainMenu = Menu(window) # 메인메뉴
window.config(menu=mainMenu)

fileMenu = Menu(mainMenu)
mainMenu.add_cascade(label='파일', menu=fileMenu)
fileMenu.add_command(label='열기', command=openImage)
fileMenu.add_command(label='저장', command=saveImage)
fileMenu.add_separator()
fileMenu.add_command(label='종료', command=shutDown)

# 일반 메뉴
photoMenu = Menu(mainMenu)
mainMenu.add_cascade(label='영상처리', menu=photoMenu)
photoMenu.add_command(label='동일 이미지', command=equal_image)
photoMenu.add_command(label='밝게 하기', command=add_image)
photoMenu.add_command(label='그레이스케일', command=grayscale_image)
photoMenu.add_command(label='이진화', command=bw_image)
photoMenu.add_command(label='반전 이미지(마우스 선택)', command=reverse_image)
photoMenu.add_command(label='채도 변경', command=change_satur)
photoMenu.add_separator()
photoMenu.add_command(label='확대', command=sizeUp)
photoMenu.add_command(label='축소', command=sizeDown)
photoMenu.add_command(label='회전', command=rotate)
photoMenu.add_separator()
photoMenu.add_command(label='엠보싱', command=emboss)
photoMenu.add_command(label='블러링', command=blur)
photoMenu.add_command(label='샤프닝', command=sharp)
photoMenu.add_separator()
photoMenu.add_command(label='침식', command=erosion)
photoMenu.add_command(label='팽창', command=dilation)
photoMenu.add_separator()
photoMenu.add_command(label='히스토그램 그래프', command=histoGraph)
photoMenu.add_command(label='스트레칭', command=stretching)
photoMenu.add_command(label='엔드 인 탐색', command=endInSearch)
photoMenu.add_command(label='평활화', command=equalized)

# OpenCV 메뉴
cvMenu = Menu(mainMenu)
mainMenu.add_cascade(label='OpenCV', menu=cvMenu)
# 화소점 처리
cvMenu.add_command(label='동일 이미지', command=equal_imageCV)
cvMenu.add_command(label='이진화', command=bwCV)
cvMenu.add_command(label='적응형 이진화', command=adaptiveBwCV)
cvMenu.add_command(label='밝게 하기', command=None) # add_imageCV
cvMenu.add_command(label='그레이스케일', command=grayscale_imageCV)
cvMenu.add_command(label='카툰', command=cartoon_imageCV)
cvMenu.add_separator() # 기하학 처리
cvMenu.add_command(label='확대', command=sizeUpCV)
cvMenu.add_command(label='축소', command=sizeDownCV)
cvMenu.add_command(label='회전', command=rotateCV)
cvMenu.add_separator() # 영역 처리
cvMenu.add_command(label='엠보싱', command=emboss_CV)
cvMenu.add_command(label='블러', command=blur_CV)
cvMenu.add_command(label='가우시안 블러', command=gaussianBlurCV)
cvMenu.add_command(label='미디안 블러', command=medianBlurCV)
cvMenu.add_command(label='양방향 블러', command=bilateralFilterCV)
cvMenu.add_separator() # 모폴로지
cvMenu.add_command(label='Erosion', command=erosionCV)
cvMenu.add_command(label='Dilation', command=dilationCV)
cvMenu.add_separator() # 추출 및 검출
cvMenu.add_command(label='윤곽선 검출', command=contoursCV)
cvMenu.add_command(label='코너 검출', command=findCornerCV)
cvMenu.add_command(label='Convex-Hull', command=convexHullCV)
cvMenu.add_command(label='색 추출', command=colorExtractionCV)
cvMenu.add_command(label='원 검출', command=findCircleCV)
cvMenu.add_separator() # 머신러닝 haarcascade
cvMenu.add_command(label='얼굴 + 눈 + 코 인식', command=face_imageCV)
cvMenu.add_command(label='고양이 얼굴 인식', command=catFace_imageCV)
cvMenu.add_command(label='성별 + 나이 인식', command=age_imageCV)
cvMenu.add_command(label='캠으로 나의 성별 나이 인식', command=ageVideo_imageCV)

window.mainloop()
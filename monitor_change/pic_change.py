import os

import cv2

'''
检测摄像头是否发现变化
https://blog.csdn.net/weixin_42456822/article/details/101131200

'''

def isPicChanged(dividePar, pointDelta, judgeh):
    '''
    通过前后帧对比，判断画面是否改变
    :param dividePar = 4 # 对比隔点，减少计算量
    :param pointDelta = 50 # 像素点的差异大于该值认为是差异点
    :param judgeh = 64 # 判断变化画面大小的阈值；画面(1/judgeTh)
    '''

    capIdx = 0  # 截图命名
    camIdx = -1
    while (int(camIdx) < 0 or int(camIdx) > 10):
        print("enter camera index in 0 and 10:")
        camIdx = int(input())

    if not (os.path.isdir('cap')):  # 创建存放截图的文件夹
        os.system('mkdir -p {}'.format("cap"))

    cap = cv2.VideoCapture(camIdx)  # 调整参数实现读取视频或者调用摄像头
    ret, frameBak = cap.read()
    for i in range(10):  # 刚打开相机时，曝光不稳定，清理10张
        ret, frameBak = cap.read()
    frame = frameBak
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frameWidth:{},frameHeight:{}".format(frameWidth, frameHeight))
    if frameWidth == 0:
        exit("camera is not available")

    while True:
        absCnt = 0
        frameBak = frame
        ret, frame = cap.read()

        for wIdx in range(int(frameWidth / dividePar)):
            for hIdx in range(int(frameHeight / dividePar)):
                if abs(int(frameBak[hIdx * dividePar][wIdx * dividePar][2]) - int(
                        frame[hIdx * dividePar][wIdx * dividePar][2])) > pointDelta:
                    absCnt += 1
        cv2.imshow("cap", frame)

        if absCnt > (frameWidth * frameHeight) / (dividePar * dividePar) / (judgeh * judgeh):
            capIdx += 2
            cv2.imwrite('cap/cap_{}.jpg'.format(capIdx), frame)
            cv2.imwrite('cap/cap_{}.jpg'.format(capIdx + 1), frameBak)
            print("get a pic:{}".format(capIdx / 2))

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":  # 这里可以判断，当前文件是否直接被python调用执行;
    isPicChanged(4, 50, 64)

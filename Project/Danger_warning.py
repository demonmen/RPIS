import os
import cv2
from base_camera import BaseCamera
import numpy as np

def Target_monitoring(img):
    import os
    import cv2
    import numpy as np
    import cv2 as cv
    yolo_dir = 'venv/cfg'  # YOLO文件路径
    weightsPath = os.path.join(yolo_dir, 'yolov3.weights')  # 权重文件
    configPath = os.path.join(yolo_dir, 'yolov3.cfg')  # 配置文件
    labelsPath = os.path.join(yolo_dir, 'coco.names')  # label名称


    CONFIDENCE = 0.5  # 过滤弱检测的最小概率
    THRESHOLD = 0.4  # 非最大值抑制阈值

    # 加载网络、配置权重
    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)  # #  利用下载的文件


    # 加载图片、转为blob格式、送入网络输入层
    #img = cv.imread(imgPath)#测试图像

    blobImg = cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), None, True, False)   # # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
    net.setInput(blobImg)  # # 调用setInput函数将图片送入输入层
    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    outInfo = net.getUnconnectedOutLayersNames()  # # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。

    # 拿到图片尺寸
    (H, W) = img.shape[:2]
    # 过滤layerOutputs
    # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
    # 过滤后的结果放入：
    boxes = [] # 所有边界框（各层结果放一起）
    confidences = [] # 所有置信度
    classIDs = [] # 所有分类ID

    # # 1）过滤掉置信度低的框框
    for out in layerOutputs:  # 各个输出层
        for detection in out:  # 各个框框
            # 拿到置信度
            scores = detection[5:]  # 各个类别的置信度
            classID = np.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
    idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD) # boxes中，保留的box的索引index存入idxs
    # 得到labels列表
    with open(labelsPath, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    # 应用检测结果
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
    if len(idxs) > 0:
        for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
            if(labels[classIDs[i]] == 'person'):
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv.rectangle(img, (x, y), (x+w, y+h), color, 2)  # 线条粗细为2px
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px
                return img,x,y,w,h
    return img,0,0,0,0
sdThresh = 10
font = cv2.FONT_HERSHEY_SIMPLEX
def distMap(frame1, frame2):
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + \
             diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist



class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        cap = cv2.VideoCapture(Camera.video_source)
        if not cap.isOpened():
            raise RuntimeError('Could not start camera.')
        _, frame1 = cap.read()
        _, frame2 = cap.read()

        while True:
            _, frame3 = cap.read()
            rows, cols, _ = np.shape(frame3)
            dist = distMap(frame1, frame3)
        
            frame1 = frame2
            frame2 = frame3
        
            mod = cv2.GaussianBlur(dist, (9,9), 0)
            _, thresh = cv2.threshold(mod, 100, 255, 0)
            _, stDev = cv2.meanStdDev(mod)
           
            if stDev > 2*sdThresh:
                cv2.imwrite('doing.jpg',mod)
                img = cv2.imread('doing.jpg')
                target,x,y,w,h = Target_monitoring(img)
                if(x==y==w==h==0):
                    yield cv2.imencode('.jpg', cv2.resize(frame2, (int(frame2.shape[1]/1.5),int(frame2.shape[0]/1.5))))[1].tobytes()
                    continue
                cv2.imwrite('target.jpg',target)
                focalLength = 557.1428571428571
                know_width = 19
                def find_marker(filename):
                    image = cv2.imread(filename)
                    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    gaussian_img = cv2.GaussianBlur(gray_img,(5,5),0)
                    edged_img = cv2.Canny(gaussian_img,35,125)
                    countours,hierarchy = cv2.findContours(edged_img.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    c = max(countours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(c)
                    return rect
                def distance_to_camera(knowWidth,focalLength,perWidth):
                    return knowWidth * focalLength/perWidth
                def cal_distance(filename,focalLength):
                    marker = find_marker(filename)
                    distance = distance_to_camera(know_width,focalLength,marker[1][0])
                    print('距离为：',distance * 2.54)
                    cv2.putText(frame2, "distance:"+str(distance * 2.54), (50, 400), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
                    return distance*2.54
                distance = cal_distance('target.jpg',focalLength)
                if distance<60:
                    import winsound
                    duration = 1000  # millisecond
                    freq = 440  # Hz
                    winsound.Beep(freq, duration)
        
                os.remove('doing.jpg')
                os.remove('target.jpg')
        
            #cv2.imshow('frame', frame2)
            yield cv2.imencode('.jpg', cv2.resize(frame2, (int(frame2.shape[1]/1.5),int(frame2.shape[0]/1.5))))[1].tobytes()

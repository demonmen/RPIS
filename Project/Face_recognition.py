import os
import cv2
from base_camera import BaseCamera
import dlib,glob
import numpy as np
from skimage import io
import operator

def send_email(img_path,to_list):#发送图片路径，收件人
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    import smtplib

    # 以html格式构建邮件内容
    send_str = '<html><body>'
    send_str += '<center>危险预警-陌生人</center>'
    send_str += '</body></html>'
    pic_path = 'test.png'
    msg = MIMEMultipart()

    # 添加邮件内容
    content = MIMEText(send_str, _subtype='html', _charset='utf8')
    msg.attach(content)

    # 邮件主题
    msg['Subject'] = '陌生人到访'

    # 邮件收、发件人
    user = "2743584180@qq.com"
    msg['To'] = ';'.join(to_list)
    msg['From'] = user
    # 密码（有些邮件服务商在三方登录时需要使用授权码而非密码，比如网易和QQ邮箱）
    passwd = "rvvawobyucebdgfg"

    # 构建并添加附件对象
    # 如果不想直接在邮件中展示图片，可以以附件形式添加
    img = MIMEImage(open(img_path, 'rb').read(), _subtype='octet-stream')
    img.add_header('Content-Disposition', 'attachment', filename=pic_path)
    msg.attach(img)

    server = smtplib.SMTP_SSL("smtp.qq.com",port=465)
    server.login(user, passwd)
    server.sendmail(user, to_list, msg.as_string())

def distMap(frame1, frame2):
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + \
             diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)

    mod = cv2.GaussianBlur(dist, (9,9), 0)
    _, thresh = cv2.threshold(mod, 100, 255, 0)
    _, stDev = cv2.meanStdDev(mod)
    return stDev

def get_images_and_labels(faces_folder_path,img_path):#人脸文件夹,需识别的人脸
    # 人脸描述子list
    descriptors = []
    # 对文件夹下的每一个人脸进行:
    # 1.人脸检测
    # 2.关键点检测
    # 3.描述子提取
    candidate = []
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        candidate.append(f.split('\\')[1])
        img = io.imread(f)
        if(len(img.shape) != 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # 1.人脸检测
        dets = detector(img, 1)

        for k, d in enumerate(dets): 
        # 2.关键点检测
            shape = sp(img, d)
        # 3.描述子提取，128D向量
            face_descriptor = facerec.compute_face_descriptor(img, shape)
        # 转换为numpy array
            v = np.array(face_descriptor)  
            descriptors.append(v)

    # 对需识别人脸进行同样处理
    # 提取描述子
    img = io.imread(img_path)
    if(len(img.shape) != 3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    dets = detector(img, 1)
    if(len(dets)==0):
        return False
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = np.array(face_descriptor) 


    # 计算欧式距离
    dist = []
    for i in descriptors:
        dist_ = np.linalg.norm(i-d_test)
        dist.append(dist_)
    return 1-np.mean(np.array(dist))#正确率




 # 1.人脸关键点检测器
predictor_path = 'venv/face-model/shape_predictor_68_face_landmarks.dat'
# 2.人脸识别模型
face_rec_model_path = 'venv/face-model/dlib_face_recognition_resnet_model_v1.dat'

# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()
# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

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

        while True:
            ret,frame = cap.read()
            ret2,frame2=cap.read()
            #cv2.imshow('camera',frame)
            yield cv2.imencode('.jpg', cv2.resize(frame, (int(frame.shape[1]/1.5),int(frame.shape[0]/1.5))))[1].tobytes()
            #动态检测
            stDev = distMap(frame, frame2)
        
            if stDev < 4:
                gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                font = cv2.FONT_HERSHEY_SIMPLEX
        
                faces = detector(gray)
                if(len(faces)):
                    for face in faces:
                        left = face.left()
                        top = face.top()
                        right = face.right()
                        bottom = face.bottom()
                        if(left<0 or top<0 or right<0 or bottom <0):
                            continue
                        imgg = np.copy(gray)
                        imgg = imgg[top:bottom,left:right]
                        cv2.imwrite('face.jpg',imgg)
                        cv2.rectangle(frame, pt1=(left, top), pt2=(right, bottom), color=(0, 255, 0), thickness=4)
        
                if(os.path.exists('face.jpg')):
        
                    test = cv2.GaussianBlur(frame, (15,15), 0)
                    cv2.putText(test,'Testing', (50, 100), font, 2.5, (255, 255, 255), 2)
                    #cv2.imshow('camera',test)
                    yield cv2.imencode('.jpg', cv2.resize(test, (int(test.shape[1]/1.5),int(test.shape[0]/1.5))))[1].tobytes()
                    preds = {'zhang':0,'mi':0,'wang':0}
                    preds['zhang'] = get_images_and_labels('image/zhang','face.jpg')
                    preds['mi'] = get_images_and_labels('image/mi','face.jpg')
                    preds['wang'] = get_images_and_labels('image/wang','face.jpg')
                    preds = sorted(preds.items(),key = operator.itemgetter(1),reverse = True)
                    if(preds[0][1]>0.65):
                        T = cv2.GaussianBlur(frame, (15,15), 0)
                        cv2.putText(T,str(preds[0][0])+': Passing', (50, 100), font, 2.5, (255, 255, 255), 2)
                        print(str(preds[0][0])+': Passing')
                        #cv2.imshow('camera',T)
                        #frame = T
                        yield cv2.imencode('.jpg', cv2.resize(T, (int(T.shape[1]/1.5),int(T.shape[0]/1.5))))[1].tobytes()
                        #time.sleep(200)
                        os.remove('face.jpg')
                        #break
                    else:
                        T = cv2.GaussianBlur(frame, (15,15), 0)
                        cv2.putText(T,'unknow', (50, 100), font, 2.5, (255, 255, 255), 2)
                        print('unknow')
                        yield cv2.imencode('.jpg', cv2.resize(T, (int(T.shape[1]/1.5),int(T.shape[0]/1.5))))[1].tobytes()
                        #time.sleep(200)
                        send_email('face.jpg','2743584180@qq.com')
                        os.remove('face.jpg')
                        import winsound
                        duration = 3000  # millisecond
                        freq = 440  # Hz
                        winsound.Beep(freq, duration)

            # encode as a jpeg image and return it
            #yield cv2.imencode('.jpg', frame)[1].tobytes()
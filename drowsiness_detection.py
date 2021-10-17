import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL
import io
import html
import time

# function to convert the JavaScript object into an OpenCV image
# 導入google JS>>return img(cv2)
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
# 建立影像對話框>>return bbox_bytes(python img type base64)
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes



  # initialize the Haar Cascade face detection model

# JavaScript to properly create our live video stream using our webcam as input
# 使用JS啟動webcam，並且顯示影像
def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;
    
    var pendingResolve = null;
    var shutdown = false;
    
    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }
    
    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }
    
    async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);
      
      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);
           
      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "environment"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);
      
      const instruction = document.createElement('div');
      instruction.innerHTML = 
          '<span style="color: red; font-weight: bold;">' +
          'When finished, click here or on the video to stop this demo</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };
      
      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; //video.videoWidth;
      captureCanvas.height = 480; //video.videoHeight;
      window.requestAnimationFrame(onAnimationFrame);
      
      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();
      
      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }
            
      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }
      
      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;
      
      return {'create': preShow - preCreate, 
              'show': preCapture - preShow, 
              'capture': Date.now() - preCapture,
              'img': result};
    }
    ''')

  display(js)

# 建立影像框，return data
def video_frame(label, bbox):
  data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
  return data

#------------------------------------------#
#------------------------------------------#

# import the face, leye, and reye cv2 database
# 載入cv2偵測檔案
face = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_righteye_2splits.xml')

# initial out module data!!
lbl=['Close','Open']
model = load_model('models/cnnCat2.h5')
model_mouth = load_model('models/mouth.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0) #to access the camera 
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
score_yawn = [0,0,0,0,0]
thicc=2
rpred=[99]
lpred=[99]
facepred=[99]
terms = 0


# start streaming video from webcam
video_stream()
# label for video
label_html = 'Capturing...'
# initialze bounding box to empty
bbox = ''
count = 0 


while True:
    # organize the img_name, if you choose the picture version.
    terms += 1
    img_name = str(terms)+".jpg"

    js_reply = video_frame(label_html, bbox)  # 使用webcam
    if not js_reply:
        break
    # frame = cv2.imread("not_yawn\\"+img_name)   # 使用照片

    # convert JS response to OpenCV Image
    # 建立JS參數
    img = js_to_image(js_reply["img"])

    # create transparent overlay for bounding box
    # 建立外框大小
    bbox_array = np.zeros([480,640,4], dtype=np.uint8)

    # grayscale image for face detection
    # 設定灰階
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ret, frame = cap.read()   # will read each frame and we store the image in a frame variable.
                                    # ret: Bool, frame: picture
        
    height, width = 480, 640  # fix the value of height and width，舊資料
    
    # 使用灰階照片偵測五官
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))  # Face detection
    left_eye = leye.detectMultiScale(gray)  # Left eye detection
    right_eye = reye.detectMultiScale(gray)  # Right eye detection

    # Create the score area
    # 建立計分框
    cv2.rectangle(bbox_array, (0,height-50) , (200,height) , (255,255,255) , thickness=cv2.FILLED )
    cv2.line(bbox_array, (0, height-50), (200, height-50), (0, 0, 255), 2)
    cv2.rectangle(bbox_array, (0,height-110) , (200,height-50) , (255,255,255) , thickness=cv2.FILLED )

    # Create the detection frame
    # 建立偵測框
    for (x,y,w,h) in faces:
        bbox_array = cv2.rectangle(bbox_array, (x,y) , (x+w,y+h) , (0,0,255) , 2 ) #Draws rectangle for detected face.
        y=int(y+(h/3)*2)
        h//=3
        h += 40

        face_detect = img[y:y+h,x:x+w]
        count = count+1
        face_detect = cv2.cvtColor(face_detect,cv2.COLOR_BGR2GRAY)
        face_detect = cv2.resize(face_detect,(24,24))   # My model is training on 24*24 images
        face_detect = face_detect/255                   # Normalization so the model can works efficiently
        face_detect = face_detect.reshape(24,24,-1)
        face_detect = np.expand_dims(face_detect,axis=0)
        facepred = model_mouth.predict_classes(face_detect)
        if(facepred[0]==1):
            score_yawn[terms%5] = 1
        if(facepred[0]==0):
            score_yawn[terms%5] = 0
        break



    for (x,y,w,h) in right_eye:
        r_eye=img[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))  # My model is training on 24*24 images
        r_eye= r_eye/255                   # Normalization so the model can works efficiently
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=img[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'
        if(lpred[0]==0):
            lbl='Closed'
        break

    # ------------------------------------------#
                    #分數系統
    # ------------------------------------------#
    if(rpred[0]==0 and lpred[0]==0):

        score=score+1
        cv2.putText(bbox_array,"Closed",(10,height-20), font, 1,(255,0,0),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):

    else:

        score=score-1
        cv2.putText(bbox_array,"Open",(10,height-20), font, 1,(255,0,0),1,cv2.LINE_AA)

    if(score<0):
        score=0
    cv2.putText(bbox_array, 'yawn: ', (10, height - 90), font, 1, (255,0,0), 1, cv2.LINE_AA)
    cv2.putText(bbox_array, str(score_yawn), (10, height - 65), font, 1, (255,0,0), 1, cv2.LINE_AA)   
    cv2.putText(bbox_array,'Score:'+str(score),(100,height-20), font, 1,(255,0,0),1,cv2.LINE_AA)
    
    yawn_alarm = [1,1,1,1,1]
    yawn_alarm = True if score_yawn == yawn_alarm else yawn_alarm
    if(score>15 or yawn_alarm==True):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),bbox_array)
        # try:
        #     sound.play()
            
        # except:  # isplaying = False
        #     pass
        if(thicc<16 ):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(bbox_array,(0,0),(width,height),(255,0,0),thicc) 

    # ------------------------------------------#
                    # 影像組裝
    # ------------------------------------------#
    bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255

    # convert overlay of bbox into bytes
    # 丟進function組合影像
    bbox_bytes = bbox_to_bytes(bbox_array)

    # update bbox so next frame gets new overlay
    bbox = bbox_bytes


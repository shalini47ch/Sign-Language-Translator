import cv2
import numpy as np
from gtts import gTTS

def nothing(x):
    pass

image_x, image_y = 64,64

from tensorflow.keras.models import load_model
classifier = load_model('my_model1.h5')
 def fetch():
    language='en'
    myob=gTTS(text=re_text,lang=language,slow=False)
    myob.save('Voice.mp3')

def predictor():
       import numpy as np
       from tensorflow.keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'A'
       elif result[0][1] == 1:
              return 'B'
       elif result[0][2] == 1:
              return 'C'
       elif result[0][3] == 1:
              return 'D'
       elif result[0][4] == 1:
              return 'E'
       elif result[0][5] == 1:
              return 'F'
       elif result[0][6] == 1:
              return 'G'
       elif result[0][7] == 1:
              return 'H'
       elif result[0][8] == 1:
              return 'I'
       elif result[0][9] == 1:
              return 'J'
       elif result[0][10] == 1:
              return 'K'
       elif result[0][11] == 1:
              return 'L'
       elif result[0][12] == 1:
              return 'M'
       elif result[0][13] == 1:
              return 'N'
       elif result[0][14] == 1:
              return 'O'
       elif result[0][15] == 1:
              return 'P'
       elif result[0][16] == 1:
              return 'Q'
       elif result[0][17] == 1:
              return 'R'
       elif result[0][18] == 1:
              return 'S'
       elif result[0][19] == 1:
              return 'T'
       elif result[0][20] == 1:
              return 'U'
       elif result[0][21] == 1:
              return 'V'
       elif result[0][22] == 1:
              return 'W'
       elif result[0][23] == 1:
              return 'X'
       elif result[0][24] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'Z'
       

       

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
re_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)

    img = cv2.rectangle(frame, (195, 255), (465, 405), (0, 0, 0), thickness=2, lineType=8, shift=0)
    #img = cv2.rectangle(frame, (5,100),(725,400), (255,0,0), thickness=0, lineType=2, shift=0)

    lower_green = np.array([34, 177, 76]) 
    upper_green = np.array([255, 255, 255])

   
    imcrop = img[257:403, 197:463]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
   # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    masking = cv2.inRange(hsv, lower_green, upper_green)
    
    cv2.putText(frame, re_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", masking)
    
        
    img_name = "1.png"
    save_img = cv2.resize(masking, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    #print("{} written!".format(img_name))
    img_text = predictor()
    
    pre_text = img_text
   
    print(pre_text)
    k= cv2.waitKey(1)
    if k == 32:
        re_text = re_text+" "
    elif k == ord('k'):
        re_text = re_text+img_text
        fetch()
    elif k == 27:
        break


cam.release()
cv2.destroyAllWindows()

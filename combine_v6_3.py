import cv2 as cv
import numpy as np
import math
cap = cv.VideoCapture(0)

faceCas = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
smileCas = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

def smiledetect(gray, frame,newImageCopy):
    faces = faceCas.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30),)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        r_gray = gray[y:y + h, x:x + w]
        r_color = frame[y:y + h, x:x + w]
		
        smiles = smileCas.detectMultiScale(r_gray, scaleFactor = 1.5, minNeighbors = 20, minSize = (20,20),)
        for(sx, sy, sw, sh) in smiles:
            cv.rectangle(r_color, (sx,sy), ((sx + sw), (sy + sh)), (0, 255, 0), 2)
            cv.imwrite("result.png", newImageCopy)
    return frame

decision = int(input("Please select mode: 1. smile detect 2. Gesture detect \n"))

while(1):
    try:
        if decision == 1:
            #read frame from videoCapture
            _, img = cap.read()
            img = cv.flip(img,1)
            newImageCopy = img.copy()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            #to detect if there is face with smile
            canvas = smiledetect(gray,img,newImageCopy)
            cv.imshow("Smile detection", canvas)
		
        elif decision ==2:        
            ret, frame = cap.read()
            frame=cv.flip(frame,1)
            newImageCopy = frame.copy()
            kernel = np.ones((3,3),np.uint8)
            
            #to set the detection area
            detection_area = frame[100:300, 100:300]        
            cv.rectangle(frame,(100,100),(300,300),(0,255,0),0) 
            
            #to set the skin color range, by using HSV
            #hsv = cv.cvtColor(detection_area, cv.COLOR_BGR2HSV)  
            #lower_skin = np.array([0,20,70], dtype=np.uint8)
            #upper_skin = np.array([20,255,255], dtype=np.uint8) 
            
            #extrapolate the hand to fill dark spots within
            yCrCb = cv.cvtColor(detection_area, cv.COLOR_BGR2YCR_CB)
            (_,cr,_) = cv.split(yCrCb)
            cr1 = cv.GaussianBlur(cr, (5,5),0)
            _,skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            mask = skin
            mask = cv.dilate(mask, kernel, iterations = 4)
        
            #to find the contours
            contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
            #to find the contour of max area of hand
            contour_of_hand = max(contours, key = lambda x: cv.contourArea(x))
        
            #to approximate the contour a little
            epsilon = 0.0005*cv.arcLength(contour_of_hand,True)
            approx= cv.approxPolyDP(contour_of_hand,epsilon,True)       
        
            #to make the convex hull around hand
            hull = cv.convexHull(contour_of_hand)
        
            #to set the area of the hull
            areahull = cv.contourArea(hull)
            
            #to set the area of the hand
            areacnt = cv.contourArea(contour_of_hand)
      
            #to find the percentage of area that not covered by hand in the convex hull
            arearatio = ((areahull - areacnt) / areacnt) * 100
        
            #to find the defects in convex hull with respect to hand        
            hull = cv.convexHull(approx, returnPoints=False)         
            defects = cv.convexityDefects(approx, hull)
        
        
            # l = no. of defects
            l = 0
        
            #to find the number of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt= (100,180)            
            
                #to find length of all three sides of the triangle                
                a = math.sqrt(pow((end[0] - start[0]),2) + pow((end[1] - start[1]), 2))
                b = math.sqrt(pow((far[0] - start[0]), 2) + pow((far[1] - start[1]), 2))
                c = math.sqrt(pow((end[0] - far[0]), 2) + pow((end[1] - far[1]), 2))
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
            
                #apply cosine rule here
                angle = math.acos((pow(b, 2) + pow(c, 2) - pow(a, 2)) / (2 * b * c)) * 57
            
        
                #ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                #(2 * ar) / a is the distance between point and convex hull
                if angle <= 90 and (2 * ar) / a > 30:
                    l += 1
                    cv.circle(detection_area, far, 3, [255,0,0], -1)
            
                #draw lines around hand
                cv.line(detection_area,start, end, [0,255,0], 2)
            
            
            #l+=1
        
            #print corresponding gestures which are in their ranges
            font = cv.FONT_HERSHEY_SIMPLEX
            if l == 0:            
                if areacnt < 5000:
                    cv.putText(frame,'Put hand in the box',(0,50), font, 2, (255,255,255), 2, cv.LINE_AA)
                    
                else:
                    if arearatio<12:
                        cv.putText(frame,'0',(0,50), font, 2, (0,0,0), 2, cv.LINE_AA)
                    elif arearatio<17.5:
                        cv.putText(frame,'Good job',(0,50), font, 2, (255,255,255), 2, cv.LINE_AA)
                   
                    else:
                        cv.putText(frame,'1',(0,50), font, 2, (255,255,255), 2, cv.LINE_AA)
                    
            elif l == 1:
                cv.putText(frame,'V sign',(0,50), font, 2, (255,255,255), 2, cv.LINE_AA)
                #saving image when V sign
                cv.imwrite("result.png", newImageCopy)            
        
            elif l == 2:
                if arearatio<27:
                    cv.putText(frame,'3 sign',(0,50), font, 2, (255,255,255), 2, cv.LINE_AA)
                else:
                    cv.putText(frame,'ok',(0,50), font, 2, (255,255,255), 2, cv.LINE_AA)
                    
            elif l == 3:
                cv.putText(frame,'4',(0,50), font, 2, (255,255,255), 2, cv.LINE_AA)
            
            elif l == 4:
                cv.putText(frame,'5',(0,50), font, 2, (255,255,255), 2, cv.LINE_AA)
            
            else :
                cv.putText(frame,'reposition',(10,50), font, 2, (255,255,255), 2, cv.LINE_AA)
            
            #to show the windows
            cv.imshow('mask',mask)        
            cv.imshow('frame',frame) 
        else:
            print("invalid input")
    except:
        #print("No output")
        pass
      
    if cv.waitKey(1) & 0xff == ord('q'):
        break
    
cv.destroyAllWindows()
cap.release()    
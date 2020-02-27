import cv2
import matplotlib.pyplot as plt 
import numpy as np


cap = cv2.VideoCapture(0) #change for external cameras

def preprocessing(mask):
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    (cnts,_) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


while(True):
    # capture frame-by-frame
    _, img = cap.read()

    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # Range for lower red
    lower_red = np.array([50,120,150])
    upper_red = np.array([60,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # # Range for upper range
    lower_red = np.array([170,120,150])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    
    # # Generating the final mask to detect red color
    red_mask = mask1+mask2

    # # Laterally invert the image / flip the image
    # img  = np.flip(img, axis=1)
    
    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # Range for lower green
    lower_red = np.array([40,120,150])
    upper_red = np.array([70,255,255])
    green_mask = cv2.inRange(hsv, lower_red, upper_red)

    cnts_g = preprocessing(green_mask)
    cnts_r = preprocessing(red_mask)

    # loop over the contours
    for c in cnts_g:
        area = cv2.contourArea(c)
        if(area > 100):
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) )
            cY = int((M["m01"] / M["m00"]) )
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c = c.astype("int")
            cv2.drawContours(img, [c], -1, (0, 0, 0), 2)
            cv2.putText(img, str(area), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
            cv2.putText(img, str(cY), (cX + 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,255,0),2)
        
    for c in cnts_r:
        area = cv2.contourArea(c)
        if(area > 100):
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) )
            cY = int((M["m01"] / M["m00"]) )
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c = c.astype("int")
            cv2.drawContours(img, [c], -1, (0, 0, 0), 2)
            cv2.putText(img, str(area), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
            cv2.putText(img, str(cY), (cX + 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(255,0,0),2)


    # display the processed frame
    cv2.imshow("tracking", img)
    
    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
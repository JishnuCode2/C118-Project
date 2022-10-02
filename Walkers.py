import cv2


# Create our body classifier
fullbody_cascade = cv2.CascadeClassifier('C:/Users/JISHNU D/Downloads/PRO-118-Project/haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('C:/Users/JISHNU D/Downloads/PRO-118-Project/walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    people = fullbody_cascade.detectMultiScale(gray,1.2,5)    
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in people:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),4)

    #Show video
    cv2.imshow("Video",frame)
    if cv2.waitKey(50) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()

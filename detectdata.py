import cv2,numpy,os
harr='haarcascade_frontalface_default.xml'
datasets='dataset'
print('Training...')
(images,labels,names, id)=([],[],{},0)
for(subdir,dir,files) in os.walk(datasets):
    for subdir in dir:
        names[id]=subdir
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1
(width,heigth)=(130,100)
(images,labels)=[numpy.array(lis) for  lis in [images,labels]]
model = cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)
face_cascade=cv2.CascadeClassifier(harr)
webcam=cv2.VideoCapture(0)
cnt=0
while True:
    (a, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # coordinates of face
    for (x,y,w,h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, heigth))
        predition=model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        if predition[1]<800:
            cv2.putText(im,'%s-%.0f'%(names[predition[0]],predition[1]),(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(51,255,255))
            print(predition[0])
            cnt=0
        else:
            cnt+=1
            cv2.putText(im, 'Unknown' , (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0))
            if(cnt>100):
                print("Unknown person")
                cv2.imwrite("input.jpg",im)
                cnt=0
    cv2.imshow('opencv',im)
    key=cv2.waitKey(10)
    if key=='q':
        break
webcam.release()
cv2.destroyAllWindows()


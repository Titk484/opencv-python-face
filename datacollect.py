import cv2

video=cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#ID Create
id = input("Nhập ID của bạn: ")
# id = int(id)
count = 0

while True:
    ret,frame = video.read()
    #Chuyển sang nền xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces :
        count = count + 1
        cv2.imwrite('datasets/User.'+ str(id) +"."+ str(count)+ ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if count>5:
        break

video.release()
cv2.destroyAllWindows()
print("Bộ dữ liệu sưu tập đã được thực hiện...")
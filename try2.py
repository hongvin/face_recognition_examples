import face_recognition
import cv2
import youtube_dl
import requests
from bs4 import BeautifulSoup
import requests

import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

###GET LECT INFO
page=requests.get("https://engine.um.edu.my/academic-staff")
soup=BeautifulSoup(page.content,"html.parser")

div_p_img=soup.select('div p img')
div_p_a=soup.select('div p a')

lect_name,pic_link=[],[]
for i in div_p_a[:]:
    if 'umexpert' not in str(i):
        div_p_a.remove(i)
    else:
        lect_name.append(i.text)

for i in div_p_img[:]:
    if ('img/files/image' not in str(i)) and ('Staff' not in str(i)):
        div_p_img.remove(i)
    else:
        pic_link.append('https://engine.um.edu.my/'+i.get('src'))
###

###YOUTUBE LINK: DESPACITO 
ydl_out={'nocheckcertificate': True,}
youtube_link="https://www.youtube.com/watch?v=CYRpv-cMqx0"
ytdl=youtube_dl.YoutubeDL(ydl_out)

info_dict = ytdl.extract_info(youtube_link, download=False,)
formats = info_dict.get('formats',None)

for f in formats:
    if f.get('format_note',None)=='360p':
        url=f.get('url',None)

###

input_video = cv2.VideoCapture(url)
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'MPEG')

output_video = cv2.VideoWriter('output.avi', fourcc, 25.07, (1280, 720))

###DOWNLOAD PICTURE
# print('***DOWNLOADING PIC***')
# i=0
# for url in pic_link:
#     print('[Downloading image] %s' %lect_name[i])
#     with open(lect_name[i]+'.jpg', 'wb') as f:
#         response = requests.get(url)
#         f.write(response.content)
#     i=i+1
# print('***FINISH DOWNLOADING PIC***')
###

###ENCODING IMAGES
print('***ENCODING IMAGES***')
print(len(lect_name))
known_faces=[]
error_loc=[]
for i in range(len(pic_link)):
    print('[Encoding] %s' %lect_name[i])
    try:
        img_rec=face_recognition.load_image_file(lect_name[i]+'.jpg')
        img_enc=face_recognition.face_encodings(img_rec)[0]
        known_faces.append(img_enc)
    except:
        print('******ERROR!!!********')
        error_loc.append(i)
print(error_loc)
for i in range(len(error_loc)):
    lect_name.pop(error_loc[i]-i)
print('***DONE***')
###

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    ret, frame = input_video.read()
    frame_number += 1

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        for i in range(len(match)):
            if match[i]:
                name=lect_name[i]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        print("[FACE FOUND] Frame %i: %s" %(frame_number,name))

    print("Writing frame {} / {}".format(frame_number, length))
    output_video.write(frame)
    # cv2.imshow('vid',frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

input_video.release()
cv2.destroyAllWindows()
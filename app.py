#Importing libraries
import torch
import io
from PIL import Image
import cv2
import numpy as np
import imutils
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from itsdangerous import Signer, BadSignature, want_bytes
import psycopg2 #installing postgresql
import psycopg2.extras
import pytesseract #extraction of text from image using tesseract

conn = psycopg2.connect( #psycopg2 database adaptor for implementing python
        host="localhost",
        database="num_plate",
        user='postgres',
        password='p@ssw0rd')

app = Flask(__name__)

#yolov5 model with the trained weights of dataset
model = torch.hub.load( '/home/neosoft/Documents/ocr/yolov5','custom',path='/home/neosoft/Documents/ocr/best.pt',source='local') # for PIL/cv2/np inputs and

#for detecting the number plate
def extraction(img_bytes):
    pic = Image.open(io.BytesIO(img_bytes))
    results = model(pic, size=640)  
    return results

#flask home page 
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file') #to take i/p(image) from the user
        if not file:
            return

        img_bytes = file.read()
        results = extraction(img_bytes)
        print(type(results))
        results.save('results0.jpg')
    
        croped_img = results.crop(save=True) #cropping the image on bounding box
        croped_img=croped_img[0]['im'] #to get the image array from the array created from above step
        img = imutils.resize(croped_img, width=300 )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #to convert image to gray
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        img = cv2.bilateralFilter(img, 11, 17, 17) 
        img = cv2.medianBlur(img, 3)
        #cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        
        img=Image.fromarray(img) #to get image from array
        img.show()#to display image
        text = pytesseract.image_to_string(img)#to extract the charcters from number plate
        
        #for removing empty spaces
        sent1='' 
        for i in text:
            if i==" ":
                continue
            else:
                sent1+=i
        sent2=''        
        for i in sent1:
            if i.lower()!=i or i.isdigit():
                sent2+=i        
        
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        s = "SELECT * FROM VEHICLE where vehicle_number =%s"

        cur.execute(s, (sent2,))
        
        res = cur.fetchall()
        if len(res)==1:
            return render_template('results.html',path=sent2,msg="Access granted for your car ...")#to redirect to results html page   

        else:
            return render_template('results.html',path=sent2,msg="Access not granted for your car ...")#to redirect to results html page   

    return render_template('index.html')    
app.secret_key = 'the random string' 
if __name__ == "__main__":
    app.run(debug=True,port=5500)

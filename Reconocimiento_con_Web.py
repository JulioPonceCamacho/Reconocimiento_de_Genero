from flask import Flask, render_template, Response, request, request_finished
import cv2
import os
import numpy as np
import json
import imutils


COLORES = [[255,100,100],[80,80,255],[167,249,49],[226,151,178],[221,151,226],[131,239,234]]
class Video():
    Contraste = 0
    DispositivoAn=0
    Dispositivo = 0
    Saturacion = 0
    Zoom = 0
    Brillo = 50
    Color = 0
    Exposicion=0
    CH=0
    CM=4
    Video=None
    def __init__(self,numero):
        self.Video=cv2.VideoCapture(numero)
    def cambiarVideo(self,numero):
        self.Video.release()
        self.Video=cv2.VideoCapture(numero)
    def cerrarVideo(self):
        self.Video.release()
    def configurarVideo(self,num,ex):
        self.Video.set(num,ex)
    def ComprobarVideo(self):
        try:
            self.leerVideo()
            return True
        except:
            print("asj")
            return False            
    def leerVideo(self):
        return self.Video.read()
    def Configurar(self,Contraste,Dispositivo,Saturacion,Zoom,Brillo,Color,Exposicion,CH,CM):
        self.Contraste=Contraste+127
        self.Dispositivo = Dispositivo
        self.Saturacion=Saturacion
        self.Zoom=Zoom
        self.Brillo=Brillo+255
        self.Color=Color
        self.Exposicion=Exposicion
        self.CH = CH
        self.CM= CM

video = Video(0)
video.Configurar(0, 0, 50,0, 0, 0, 50,0,4)
#Listas a manejar en el reconocimiento de edad
Lista_Edad=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
Lista_Genero=['Hombre','Mujer']

#Modelos y achivos necesarios que tienen los parametros para construir la red neuronal
Cara_Proto="Deteccion_Cara/opencv_face_detector.pbtxt"
Cara_Modelo="Deteccion_Cara/opencv_face_detector_uint8.pb"
Edad_Proto="Deteccion_Edad/age_deploy.prototxt"
Edad_Modelo="Deteccion_Edad/age_net.caffemodel"
Genero_Proto="Deteccion_Genero/gender_deploy.prototxt"
Genero_Modelo="Deteccion_Genero/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)


def resaltar(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    Alto_Frame=frameOpencvDnn.shape[0]
    Ancho_Frame=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.2, (300,300), [104, 117, 123], True, False)
    net.setInput(blob)
    Detecciones=net.forward()
    Cara_Rect=[]
    x1=0
    y1=0
    x2=0
    y2=0
    for i in range(Detecciones.shape[2]):
        confidence=Detecciones[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(Detecciones[0,0,i,3]*Ancho_Frame-10)
            y1=int(Detecciones[0,0,i,4]*Alto_Frame-10)
            x2=int(Detecciones[0,0,i,5]*Ancho_Frame+10)
            y2=int(Detecciones[0,0,i,6]*Alto_Frame)
            Cara_Rect.append([x1,y1,x2,y2])
    return frameOpencvDnn,Cara_Rect


Cara_Net=cv2.dnn.readNet(Cara_Modelo,Cara_Proto)
Edad_Net=cv2.dnn.readNet(Edad_Modelo,Edad_Proto)
Genero_Net=cv2.dnn.readNet(Genero_Modelo,Genero_Proto)
padding=20
saturacion=0
app = Flask(__name__)
 
@app.route('/') 
def index():
    return render_template('index.html')
    
def Zoom(img, porcentaje):
    if porcentaje==0:
        return img
    else:
        porcentaje=porcentaje/2
        w=porcentaje*img.shape[1]//100
        w=w//2
        h= porcentaje*img.shape[0]//100
        h=h//2 
        return img[int(h):int(img.shape[0]-h),int(w):int(img.shape[1]-w)]

def gen():
    while True:
        try:
            cam=0
            if video.DispositivoAn!= video.Dispositivo:
                video.cambiarVideo(video.Dispositivo)
                video.DispositivoAn=video.Dispositivo
                cam=1  
            if cam==1:
                video.ComprobarVideo()
            video.configurarVideo(10,video.Exposicion)
            hasFrame,frameOr=video.leerVideo()
            frameOr = Zoom(frameOr,video.Zoom)
            frameOr = controller(frameOr, video.Brillo, video.Contraste) 
            frame=frameOr
            fram=frameOr
            i=0
            try:
                Genero=''
                resultImg,Cara_Rect=resaltar(Cara_Net,frame)
                hombres=None 
                mujeres=None
                h=0
                m=0
                if len(Cara_Rect)>0:
                    while i<len(Cara_Rect):
                        faceBox = Cara_Rect[i]
                        cara=frameOr[faceBox[1]:faceBox[3],faceBox[0]:faceBox[2]]
                        Cara=frame[max(0,faceBox[1]-padding):
                                min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                                :min(faceBox[2]+padding, frame.shape[1]-1)]
                        blob=cv2.dnn.blobFromImage(Cara, 1.0, (237,237), MODEL_MEAN_VALUES, swapRB=False)
                        Genero_Net.setInput(blob)
                        Genero_Predic=Genero_Net.forward()
                        Genero=Lista_Genero[Genero_Predic[0].argmax()]
                        Edad_Net.setInput(blob)
                        Edad_Predic=Edad_Net.forward()
                        Edad=Lista_Edad[Edad_Predic[0].argmax()]
                        text = "{}:{}".format(Genero, Edad)
                        #Coloca el texto en la imagen
                        Area = (faceBox[2]-faceBox[0])*(faceBox[3]-faceBox[1])
                        fuente=Area*.9/40000
                        if fuente > .6 :
                            grosor=2
                        else :
                            grosor=1
                        if fuente <0.5:
                            fuente = fuente +0.2
                        val=frameOr.shape[0]//6
                        cara=cv2.resize(cara,[(frameOr.shape[0] // 6), ((frameOr.shape[0] // 6)+10)])
                        cara= imutils.resize(cara, height = (frameOr.shape[0] // 6), width= (frameOr.shape[0] // 6)+10)
                        if Genero == 'Hombre':
                            cv2.rectangle(frameOr, (faceBox[0],faceBox[1]), (faceBox[2],faceBox[3]),COLORES[video.CH], int(round(resultImg.shape[0]/150)), 8)
                            cv2.putText(frameOr, text,(faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, fuente, COLORES[video.CH], grosor, cv2.LINE_AA)
                            if h==0:
                                hombres=cara
                                h+=1
                            else:
                                hombres= cv2.vconcat([hombres, cara])
                                h+=1
                        else:
                            cv2.rectangle(frameOr, (faceBox[0],faceBox[1]), (faceBox[2],faceBox[3]), COLORES[video.CM], int(round(resultImg.shape[0]/150)), 8)
                            cv2.putText(frameOr, text,(faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX,fuente,COLORES[video.CM], grosor,cv2.LINE_AA)
                            if m==0:
                                mujeres=cara
                                m+=1
                            else:
                                mujeres= cv2.vconcat([mujeres, cara])
                                m+=1
                        frame=cv2.rectangle(resultImg,(faceBox[0],faceBox[1]), (faceBox[2],faceBox[3]), (0,0,0), -1)
                        i+=1
                            
                resultImg=frameOr
            except:
                err=0
                Genero='-'
                cara=cv2.imread("Media/ICRH.jpg")
                cara=cv2.resize(cara,[(fram.shape[0] // 6), ((fram.shape[0] // 6)+10)])
                cara= imutils.resize(cara,  height = (fram.shape[0] // 6), width= (fram.shape[0] // 6)+10)
                h=0
                if h==0:
                    hombres=cara
                    while h<5:
                        hombres=cv2.vconcat([hombres,cara])
                        h+=1
                hombres= imutils.resize(hombres, height = (fram.shape[0]))
                cara=cv2.imread("Media/ICRM.jpg")
                cara=cv2.resize(cara,[(fram.shape[0] // 6), ((fram.shape[0] // 6)+10)])
                cara= imutils.resize(cara,  height = (fram.shape[0] // 6), width= (fram.shape[0] // 6)+10)
                m=0
                if m==0:
                    mujeres=cara
                    while m<5:
                        mujeres=cv2.vconcat([mujeres,cara])
                        m+=1
                mujeres= imutils.resize(mujeres, height = (fram.shape[0]))
                #hombres=cv2.resize(hombres,[resultImg.shape[0],resultImg.shape[1]])
                hombres=cv2.hconcat([hombres,mujeres])
                res=cv2.hconcat([fram,hombres])
                ret, jpeg = cv2.imencode('.jpg', res)
                imgJPG = jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + imgJPG + b'\r\n\r\n')
            if Genero=='':
                cara=cv2.imread("Media/ICRH.jpg")
                cara=cv2.resize(cara,[(frameOr.shape[0] // 6), ((frameOr.shape[0] // 6)+10)])
                cara= imutils.resize(cara,  height = (frameOr.shape[0] // 6), width= (frameOr.shape[0] // 6)+10)
                if h==0:
                    hombres=cara
                    while h<5:
                        hombres=cv2.vconcat([hombres,cara])
                        h+=1
                hombres= imutils.resize(hombres, height = (frameOr.shape[0]))
                cara=cv2.imread("Media/ICRM.jpg")
                cara=cv2.resize(cara,[(frameOr.shape[0] // 6), ((frameOr.shape[0] // 6)+10)])
                cara= imutils.resize(cara,  height = (frameOr.shape[0] // 6), width= (frameOr.shape[0] // 6)+10)
                if m==0:
                    mujeres=cara
                    while m<5:
                        mujeres=cv2.vconcat([mujeres,cara])
                        m+=1
                mujeres= imutils.resize(mujeres, height = (frameOr.shape[0]))
                
                #hombres=cv2.resize(hombres,[resultImg.shape[0],resultImg.shape[1]])
                hombres=cv2.hconcat([hombres,mujeres])
                res=cv2.hconcat([frame,hombres])
                ret, jpeg = cv2.imencode('.jpg', res)
                imgJPG = jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + imgJPG + b'\r\n\r\n')
            elif Genero!='-':
                #Muestra la imagend e la camara Web
                try:
                    cara=cv2.imread("Media/ICRH.jpg")
                    cara=cv2.resize(cara,[(frameOr.shape[0] // 6), ((frameOr.shape[0] // 6)+10)])
                    cara= imutils.resize(cara, height = (frameOr.shape[0] // 6), width= (frameOr.shape[0] // 6)+10)
                    if h==0:
                        hombres=cara
                        while h<5:
                            hombres=cv2.vconcat([hombres,cara])
                            h+=1
                    elif h<6:
                        n=h
                        while n < 6:
                            hombres=cv2.vconcat([hombres,cara])
                            n+=1
                    hombres= imutils.resize(hombres, height = (frameOr.shape[0]))
                    cara=cv2.imread("Media/ICRM.jpg")
                    cara=cv2.resize(cara,[(frameOr.shape[0] // 6), ((frameOr.shape[0] // 6)+10)])
                    cara= imutils.resize(cara, height = (frameOr.shape[0] // 6), width= (frameOr.shape[0] // 6)+10)
                    if m==0:
                        mujeres=cara
                        while m<5:
                            mujeres=cv2.vconcat([mujeres,cara])
                            m+=1
                    elif m<6:
                        n=m
                        while n < 6:
                            mujeres=cv2.vconcat([mujeres,cara])
                            n+=1
                    mujeres= imutils.resize(mujeres, height = (frameOr.shape[0]))
                    
                    #hombres=cv2.resize(hombres,[resultImg.shape[0],resultImg.shape[1]])
                    hombres=cv2.hconcat([hombres,mujeres])
                    hombres=cv2.hconcat([resultImg,hombres])

                    ret, jpeg = cv2.imencode('.jpg', hombres)
                    imgJPG = jpeg.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + imgJPG + b'\r\n\r\n')
                except:
                    video.cerrarVideo()                
        except:
            video.cerrarVideo()
            imagen = cv2.imread("static/ups.jpg")
            ret, jpeg = cv2.imencode('.jpg', imagen)
            imgJPG = jpeg.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + imgJPG + b'\r\n\r\n')
    video.cerrarVideo()
def controller(img, brightness=255,contrast=127): 
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255)) 
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127)) 
    if brightness != 0: 
        if brightness > 0: 
            shadow = brightness 
            max = 255
        else: 
            shadow = 0
            max = 255 + brightness 
        al_pha = (max - shadow) / 255
        ga_mma = shadow 
        cal = cv2.addWeighted(img, al_pha,  
                              img, 0, ga_mma) 
    else: 
        cal = img 
    if contrast != 0: 
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast)) 
        Gamma = 127 * (1 - Alpha) 
        cal = cv2.addWeighted(cal, Alpha,  
                              cal, 0, Gamma) 
    return cal 

def duplicados(lista1,lista2,val1,val2):
    i=0
    #print("x="+str(val1)+"y="+str(val2))
    if(len(lista1)>1):
        while i<len(lista1):
            if (lista1[i]==val1 & lista2[i]==val2) | ((val1>lista1[i]+150 & val1<lista1[i]-150) & (val2>lista2[i]+150 & val2<lista2[i]-150)) :
                print("Coordenadas:"+str(val1)+","+str(val2)+" Rango: "+str(lista1[i]-100)+"-"+str(lista1[i]+100)+","+str(lista2[i]-100)+"-"+str(lista2[i]+100))
                print("Zona concreta")
                return True
            i+=1
    return False

@app.route('/test', methods=['POST'])
def test():
    output = request.get_json()
    result = json.loads(output) #this converts the json output to a python dictionary
    print(result) # Printing the new dictionary
    video.Configurar(int(result['CONT']),int(result['DS']), int(result['SV']), int(result['B']), int(result['BR']),int(result['R']), int(result['EX']),int(result['CH']), int(result['CM']))
    return result

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

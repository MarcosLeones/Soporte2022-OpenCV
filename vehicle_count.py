import cv2
import csv
import collections
import numpy as np
from tracker import *
import sqlite3
from sqlite3 import Error
import datetime

# Inicializa Tracker
tracker = EuclideanDistTracker()

# Inicializar video
#video_file_name = 'video1.mp4'
video_file_name = 'video2.mp4'
#video_file_name = 'video3.mp4'
cap = cv2.VideoCapture(video_file_name)
input_size = 320

# Ruta de la ase de datos
database_file = r"C:\sqlite\db\vehiclecount.db"

#Inicializar conección a BD
conn = sqlite3.connect(database_file)



# Intervalos de confianza
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Posición de la linea de detección
middle_line_position = 225   
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


# Guardar coco.names en una lista
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

# Indices de las clases a detectar (Auto, moto, colectivo, camión)
required_class_index = [2, 3, 5, 7]

detected_classNames = []

#yolo
modelConfiguration = 'yolov3.cfg'
modelWeigheights = 'yolov3.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# A cada clase se le asigna un color aleatorio
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Función que encuentra el cntro de un rectángulo
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# Listas que guardan la información de conteo de vehículos
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Función para contar los vehículos
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Encontrar el centro del rectángulo para poder detectar
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Encontrar la posición actual de vehículo
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Dibujar un punto en el medio del rectángulo.
    # Los vehículos se cuentan cuando el centro pasa por la línea del medio
    cv2.circle(img, center, 2, (0, 0, 255), -1) 

# Función para identificar objetos detectados
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))


    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)

    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
       
    
        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Agrega el nivel de confianza para la clase
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Dibuja el rectángulo alrededor del vehículo
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Actualizar tracker para cada objeto
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


#Define la tabla donde se van a guardar los datos obtenidos
def create_table(conn):
    # La tabla guarda la cantidad de cada tipo de vehículo que se detectaron según la dirección
    # y loggea el nombre del video con el que se corrió el programa junto a la fecha y hora
    data_table = """CREATE TABLE IF NOT EXISTS vcdata (
                                        source_video_name varchar(50),
                                        direction varchar(10),
                                        car integer,
                                        motorbike integer,
                                        bus integer,
                                        truck integer,
                                        run_date timestamp
                                    );"""

    try:
        c = conn.cursor()
        c.execute(data_table)
        c.close()
    except Error as e:
        print(e)


#Función que inserta los datos registrados en la tabla
def insert_row(conn, list, direction):
    cur = conn.cursor()
    cur.execute("insert into vcdata values (?,?,?,?,?,?,?) ", (video_file_name, direction, list[0], list[1], list[2], list[3], datetime.datetime.now()))
    conn.commit()
    cur.close()


#Función que muestra el contenido de la BD con los datos obtenidos
def show_results(conn):
    cur = conn.cursor()
    cur.execute("select * from vcdata")
    rows = cur.fetchall()

    for row in rows:
        print(row)

def realTime():

    #Inicializar tabla en BD
    create_table(conn)

    while True:
        success, img = cap.read()
        #img = cv2.resize(img,(0,0),None,0.5,0.5)
        img = cv2.resize(img, (600,400))
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Setear input
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        # Agregar datos
        outputs = net.forward(outputNames)
    
        # Encuentra los objetos que devuelve
        postProcess(outputs,img)

        # Dibujar las líneas horizontales

        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        # Dibujar los contadores
        cv2.putText(img, "Mano", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Contramano", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Auto:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Moto:        "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Colectivo:    "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Camion:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Mostrar frame
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break


    # Guarda los datos en la tabla
    insert_row(conn, up_list, "up")
    insert_row(conn, down_list, "down")

    #Mostrar resultados obtenidos
    show_results(conn)

    #Cierra conexión a BD
    conn.close()

    # Cierra captura y destruye ventanas para terminar
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    realTime()

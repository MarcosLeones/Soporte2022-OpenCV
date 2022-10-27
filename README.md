# Soporte2022-OpenCV
Trabajo práctico de OpenCV con Python para la materia Soporte

El programa realizado en Python utiliza OpenCV para contar la cantidad de vehículos que circulan por una ruta.
Es capaz de reconocer de qué tipo de vehículo se trata y en qué dirección circula.
Los datos que recoge son almacenados en una base de datos.

Para correr el programa es necesario descargar los pesos y configuración de YOLO.
El archivo "yolov3.weights" se puede encontrar aquí: https://pjreddie.com/media/files/yolov3.weights
La configuración "yolov3.cfg" se puede encontrar en https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg

Los base de datos utilizada es SQLite. 
Todas las rutas (de los archivos de YOLO, la base de datos y los videos) se pueden modificar en el archivo principal "vehicle_count.py"

Se deja también un video para usar como ejemplo. 

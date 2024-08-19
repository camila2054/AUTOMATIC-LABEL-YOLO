# AUTOMATIC-LABEL-YOLO
 For yolov5 and yolov8

-Correr label-studio de manera local
-Crear un entorno para instalar dependecias necesarias
-entrar en Label-studio-ml-backend : 
     -label-studio-ml start my_ml_backend
-tomar el http:/ e ir a label-studio y en model añadir url 
-guardar y estara conectado para realizar inferencias 
--Para exponerlo a una ip publica debemos utilizar ngrok con su token respectivo, abrir el puerto que deseemos 
-ngrok http 5000
utilizar y ngrok  https://34db-186-107-13-53.ngrok-free.app -> http://localhost:5000              
-luego en model añadir esta url ->https://34db-186-107-13-53.ngrok-free.app
-subir backend label-studio-ml start my_ml_backend -p 5000 --host 0.0.0.0
-Para cambiar a yolov5 , entrar en wsgi.py 
-------------------------------------------------------

from label_studio_ml.api import init_app
#Modificar para utilizar este modelo o el otro ejemplo model = yolov5 y modelyolov8 es con yolov8 
from modelyolov8 import NewModel
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

--------------------------------------------------------
modelyolov8 <- crea la conexión con label studio he interpreta los resultados
testYolov8.py <- Toma la imagen de label realiza la inferencia y la devuelve a la pagina 
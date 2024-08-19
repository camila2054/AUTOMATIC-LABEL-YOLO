import requests
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from typing import List, Dict, Optional
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import json
import ultralytics
import torch

class NewModel(LabelStudioMLBase):
    """Modelo de backend de ML personalizado"""
    
    def setup(self):
        """Configurar cualquier parámetro de su modelo aquí"""
        self.set("model_version", "0.0.1")

        # Token de autenticación de Label Studio    
        self.LABEL_STUDIO_TOKEN = 'e40aaa832e2a549b50583c08df1bd71954d3771e'
        self.base_url = "http://137.184.227.15:8080"
        self.headers = {
            'Authorization': f'Token {self.LABEL_STUDIO_TOKEN}'
        }

        # Intentar obtener información del proyecto para validar la API Key
        try:
            response = requests.get(f'{self.base_url}/api/projects', headers=self.headers)
            response.raise_for_status()  # Lanza un error si la solicitud no tuvo éxito
            self.set('api_key', self.LABEL_STUDIO_TOKEN)
            print("API Key validada exitosamente.")
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                print("Error: Autenticación fallida. Verifique su API Key.")
            else:
                print(f"HTTP error occurred: {http_err}")  # other errors
            raise
        except requests.exceptions.RequestException as e:
            print(f"Error al validar la API Key: {e}")
            raise

        # Cargar modelo personalizado 
        self.model_path = r'C:\Users\candr\Desktop\ENTRENAMIENTO\runs\detect\swimmingmodel_23\weights\best.pt'
        try:
            self.model = ultralytics.YOLO(self.model_path)
            print(f"Modelo cargado correctamente desde {self.model_path}")
        except Exception as e:
            print(f"Error al cargar el modelo desde {self.model_path}: {e}")
            exit()

    def test_inference(self, image_url):
        # Descargar la imagen con encabezados de autorización
        try:
            response = requests.get(image_url, headers=self.headers)
            response.raise_for_status()  # Lanza un error si la solicitud no tuvo éxito
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            print(f"Error al descargar la imagen: {e}")
            return None
        except UnidentifiedImageError as e:
            print(f"Error al abrir la imagen: {e}")
            return None

        # Redimensionar la imagen si es necesario
        img = img.resize((640, 640))  # Ajustar el tamaño según sea necesario

        # Realizar la predicción
        if torch.cuda.is_available():
            self.model.to('cuda')  # Mover el modelo a la GPU si está disponible
        
        results = self.model(img)

        # Guardar resultados en un formato que Label Studio pueda comprender
        predictions_list = []
        for box in results[0].boxes:
            x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
            conf = box.conf.item()
            cls = box.cls.item()
            width = x_max - x_min
            height = y_max - y_min
            predictions_list.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x_min / img.width * 100,
                    "y": y_min / img.height * 100,
                    "width": width / img.width * 100,
                    "height": height / img.height * 100,
                    "rectanglelabels": [self.model.names[int(cls)]]
                },
                "score": conf
            })

        result_data = {
            "width": img.width,
            "height": img.height,
            "predictions": predictions_list
        }

        return result_data

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Escribe tu lógica de inferencia aquí
        :param tasks: [Tareas de Label Studio en formato JSON](https://labelstud.io/guide/task_format.html)
        :param context: [Contexto de Label Studio en formato JSON](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return model_response
            ModelResponse(predictions=predictions) con
            predictions: [Array de predicciones en formato JSON](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """

        predictions = []

        for task in tasks:
            # Obtener la URL de la imagen desde el JSON recibido
            image_url = task['data']['image']
            print("URL de la imagen:", image_url)

            # Construir la URL completa si es necesario
            full_image_url = self.base_url + image_url if not image_url.startswith('http') else image_url
            print("URL completa de la imagen:", full_image_url)

            # Llamar a la función de inferencia
            result_data = self.test_inference(full_image_url)
            if result_data is None:
                continue

            img_width = result_data.get("width")
            img_height = result_data.get("height")
            predictions_list = result_data.get("predictions", [])

            if img_width is None or img_height is None:
                print("Error: No se pudo encontrar 'width' o 'height' en los resultados")
                continue

            # Convertir los resultados al formato de Label Studio
            labels = []
            for pred in predictions_list:
                labels.append(pred)

            predictions.append({"result": labels})

        return ModelResponse(predictions=predictions)

import requests
from model import NewModel
import os
import json

# Configuración inicial
LABEL_STUDIO_URL = 'http://137.184.227.15:8080'
API_KEY = 'e40aaa832e2a549b50583c08df1bd71954d3771e'
PROJECT_ID = 21  # Cambia esto al ID de tu proyecto
LABEL_CONFIG_PATH = r'C:\Users\candr\Documents\GITHUB\AUTOMATIC-LABEL\backend_yolov8\label_config.xml'
headers = {
    'Authorization': f'Token {API_KEY}'
}

# Cargar el label_config desde el archivo XML
def load_label_config(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise FileNotFoundError(f"El archivo {path} no existe.")

def get_all_tasks_without_annotations():
    """
    Obtiene las tareas del proyecto en Label Studio que no tienen anotaciones ni predicciones.
    """
    tasks= []
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/tasks"
    page=1
    while True:

        params = {
            'completed': False,
            'page_size': 100,  # Puedes ajustar el tamaño de la página si es necesario
            'page':page,
            'ordering': 'inner_id'  # Ordena por ID para garantizar que se cubran todas las tareas
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            tasks.extend(data)
            if len(data) < 100:
                break # Si se recibieron menos de 100 tareas, hemos terminado
            page += 1
        else:
            print(f"Error al obtener tareas: {response.status_code}, {response.text}")
            break

    return filter_tasks_without_annotations_and_predictions(tasks)

def filter_tasks_without_annotations_and_predictions(tasks):
    """
    Filtra las tareas que no tienen ni anotaciones ni predicciones.
    """
    filtered_tasks = []
    
    for task in tasks:
        annotations = task.get('annotations', [])
        predictions = task.get('predictions', [])
        if len(annotations) == 0 and len(predictions) == 0:
            filtered_tasks.append(task)
    return filtered_tasks

def print_tasks(tasks,limit=5):
    """
    Imprime las primeras 10 tareas que no tienen anotaciones ni predicciones.
    """
    tasks_sorted = sorted(tasks, key=lambda x: x['inner_id'])  # Ordenar por inner_id de menor a mayor
    for task in tasks_sorted[:limit]:
        print(f"INNER ID: {task['inner_id']}")
        print(f"Image URL: {task['data']['image']}")
        print("-" * 40)

def perform_inference_on_task(task, model):
    """
    Realiza inferencia en una tarea específica y envía los resultados a Label Studio como predicciones.
    """
    inner_id = task['inner_id']
    
     # Realizar la predicción usando el método predict
    predictions = model.predict([task])
    
    if not predictions.predictions:
        print(f"No se pudo realizar la predicción para la tarea con Inner ID {inner_id}")
        return
    else:
        print(f"Predicción realizada para la tarea con Inner ID {inner_id}")
        
        # Imprimir la estructura de predictions.predictions[0] para entender cómo acceder a sus datos
        print(f"Estructura de predictions.predictions[0]: {type(predictions.predictions[0])}")
        print(predictions.predictions[0])
        
        try:
            # Asume que predictions.predictions es una lista con al menos un elemento que tiene la clave "result"
            annotation_data = {
                "result": predictions.predictions[0].result  # Cambiar el acceso según la estructura correcta
            }
            
            # Enviar las predicciones a Label Studio
            url = f"{LABEL_STUDIO_URL}/api/tasks/{task['id']}/annotations"
            response = requests.post(url, headers=headers, json=annotation_data)
            
            if response.status_code == 201:
                print(f"Predicciones enviadas a Label Studio para la tarea con Inner ID {inner_id}")
            else:
                print(f"Error al enviar predicciones: {response.status_code}, {response.text}")
                
        except (IndexError, AttributeError) as e:
            print(f"Error al formatear los datos de predicción: {e}")
            print("Datos de predicción recibidos:", predictions.predictions)
    
if __name__ == "__main__":
    
    label_config_content = load_label_config(LABEL_CONFIG_PATH)
    model = NewModel(label_config=label_config_content)
    model.setup()

    tasks = get_all_tasks_without_annotations()
    print("Tareas obtenidas:", len(tasks))
    print(f"Tareas obtenidas sin anotaciones ni predicciones: {len(tasks)}")
    print_tasks(tasks)

    for task in tasks:
        perform_inference_on_task(task, model)

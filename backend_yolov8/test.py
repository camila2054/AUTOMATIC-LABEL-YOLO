import requests

# Configuración inicial
LABEL_STUDIO_URL = 'http://137.184.227.15:8080'
API_KEY = 'e40aaa832e2a549b50583c08df1bd71954d3771e'
PROJECT_ID = 21  # Cambia esto al ID de tu proyecto
headers = {
    'Authorization': f'Token {API_KEY}'
}

def get_tasks_without_annotations():
    """
    Obtiene las tareas del proyecto en Label Studio que no tienen anotaciones ni predicciones.
    """
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/tasks"
    params = {
        'completed': False,
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        tasks = response.json()
        return filter_tasks_without_annotations_and_predictions(tasks)
    else:
        print(f"Error al obtener tareas: {response.status_code}, {response.text}")
        return []

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

def print_tasks(tasks,limit=25):
    """
    Imprime las primeras 10 tareas que no tienen anotaciones ni predicciones.
    """
    tasks_sorted = sorted(tasks, key=lambda x: x['inner_id'])  # Ordenar por inner_id de menor a mayor
    for task in tasks_sorted[:limit]:
        print(f"Inner ID: {task['inner_id']}")
        print(f"Image URL: {task['data']['image']}")
        print("-" * 40)

if __name__ == "__main__":
    tasks = get_tasks_without_annotations()
    print_tasks(tasks)

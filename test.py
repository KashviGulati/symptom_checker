import requests

url = "http://127.0.0.1:5000/predict"
data = {"symptoms": ["anxiety", "restlessness", "lethargy", "mood_swings"]}

response = requests.post(url, json=data)
print(response.json())  

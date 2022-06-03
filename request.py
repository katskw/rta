# możesz też wykorzystac kod pythona
import requests
response = requests.get("http://127.0.0.1:5000/api/v1.0/predict?&sl=4.5&pl=1.3")
print(response.content)
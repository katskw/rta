# możesz też wykorzystac kod pythona
import requests
response = requests.get("http://127.0.0.1:80/api/v1.0/predict?&x1=4.5&x2=1.3")
print(response.content)
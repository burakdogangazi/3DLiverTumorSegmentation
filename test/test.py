import requests

resp = requests.post("http://127.0.0.1:8080/predict",files={'file':open('liver_0.nii',"rb")})

print(resp.text)
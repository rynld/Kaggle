import requests

url = "http://localhost:59895/api/v1/clienteling/clients/perka552200/reward-program-summary"

payload = "{\"userName\":\"irina\",\"password\":\"irina\",\"ringingid\":\"\"}"
headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
    'postman-token': "f8c6903f-82f7-ff5c-5b0b-3983d7359e29"
    }

response = requests.request("GET", url, data=payload, headers=headers)

print(response.text)
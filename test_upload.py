import requests

# FastAPI 地址
url = "http://127.0.0.1:8000/optimize"

# 打开 CSV 文件
with open("mock_predictions.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# 打印返回结果
print(response.json())

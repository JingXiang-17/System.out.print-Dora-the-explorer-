from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from optimization import optimize_flight_delays  # 你的优化逻辑函数

app = FastAPI(title="Flight Delay Optimization API")

# ----------------------
# POST 接口: 接收 CSV 文件
@app.post("/optimize")
async def optimize_csv(file: UploadFile = File(...)):
    try:
        # 读取上传的 CSV
        df = pd.read_csv(file.file)

        # ----------------------
        # 检查必要列
        required_cols = ['flight_id', 'predicted_delay']
        for col in required_cols:
            if col not in df.columns:
                return JSONResponse(
                    content={"error": f"CSV 必须包含 {required_cols} 列"},
                    status_code=400
                )

        # ----------------------
        # 转换成优化函数需要的格式
        flights = []
        for idx, row in df.iterrows():
            flights.append({
                "flight": row["flight_id"],
                "predicted_delay": row["predicted_delay"]
            })

        # ----------------------
        # 调用优化模型
        solution = optimize_flight_delays(flights)

        # 返回 JSON
        return {"optimization_result": solution}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ----------------------
# GET 接口: 测试 API 是否正常
@app.get("/")
def read_root():
    return {"message": "Flight Delay Optimization API 正常运行"}

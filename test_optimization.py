import pandas as pd
from ortools.linear_solver import pywraplp

def optimize_flight_delays(flights):
    """
    flights: 航班列表，每个元素是字典
    {
        "flight": "AA101",
        "predicted_delay": 30  # ML预测延误分钟数
    }
    输出：
        每个航班建议增加的缓冲时间 (buffer)
    """
    # 创建优化求解器
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("SCIP solver 不可用")
        return

    # 定义决策变量: 每个航班的缓冲时间，范围 0~180 分钟
    buffer_vars = {}
    for f in flights:
        buffer_vars[f["flight"]] = solver.IntVar(0, 180, f["flight"])

    # 定义目标函数: 最小化总缓冲时间
    objective = solver.Objective()
    for f in flights:
        objective.SetCoefficient(buffer_vars[f["flight"]], 1)
    objective.SetMinimization()

    # 添加约束: 如果预测延误 > 15 分钟，则缓冲 ≥ 5
    for f in flights:
        if f["predicted_delay"] > 15:
            solver.Add(buffer_vars[f["flight"]] >= 5)

    # 求解优化问题
    status = solver.Solve()

    # 输出结果
    if status == pywraplp.Solver.OPTIMAL:
        results = {}
        for f in flights:
            results[f["flight"]] = buffer_vars[f["flight"]].solution_value()
        return results
    else:
        return "没有找到最优解"

# ----------------------
# 从 CSV 读取 ML 输出并运行优化
def run_from_csv(csv_path):
    """
    csv_path: ML 输出 CSV 路径
    CSV 必须包含两列: 'flight' 和 'predicted_delay'
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 转换成优化函数需要的格式
    flights = []
    for idx, row in df.iterrows():
        flights.append({
            "flight": row["flight"],
            "predicted_delay": row["predicted_delay"]
        })

    # 调用优化函数
    solution = optimize_flight_delays(flights)

    # 输出优化结果
    print("优化建议 (buffer 分钟):")
    for flight, buffer_time in solution.items():
        print(f"{flight}: {buffer_time} min")

    # 可选: 保存结果到 CSV
    output_df = df.copy()
    output_df["buffer_minutes"] = output_df["flight"].map(solution)
    output_df.to_csv("optimized_buffer.csv", index=False)
    print("优化结果已保存到 optimized_buffer.csv")

# ----------------------
# 测试用
if __name__ == "__main__":
    # 假设 ML 输出 CSV 路径
    csv_path = "ml_predicted_delays.csv"
    run_from_csv(csv_path)

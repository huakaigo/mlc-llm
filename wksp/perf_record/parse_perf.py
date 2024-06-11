import os, sys
import json
from typing import Dict

def _print_table(record: Dict):
    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["func", "arg shape", "run times", "total cost time(ms)", "avg cost time(ms)"]
    record_sorted = sorted(record.items(), key=lambda x: x[1][0], reverse=True)
    other_cost = 0
    for func, cost in record_sorted:
        if cost[0] < 100: # 小于100us记录到other中
            other_cost += cost[0]
            continue
        formatted_cost = "{:.6f}".format(cost[0]/1000.0)
        formatted_avg_cost = "{:.6f}".format(cost[0]/cost[1]/1000.0)
        # table.add_row([func[0], "output_shape: " + func[1].split("],")[-1], cost[1], formatted_cost, formatted_avg_cost])
        table.add_row([func[0], func[1], cost[1], formatted_cost, formatted_avg_cost])

    table.add_row(["other", "-", "-", "{:.6f}".format(other_cost/1000.0) , "-"])

    print(table)
    return table

def _main(perf_json: str):
    with open(perf_json, 'r') as f:
        data = json.loads(f.read())
    record = {}
    total = 0
    for c in data["calls"]:
        func_name = c["Name"]["string"]
        func_arg_shapes = c["Argument Shapes"]["string"]
        cost_time = float(c["Duration (us)"]["microseconds"])
        ## 如果调用耗时小于50us,则合并统计
        if cost_time < 50:
            func_arg_shapes = "all shape merged"
        if (func_name, func_arg_shapes) not in record:
            record[(func_name, func_arg_shapes)] = [0, 0]
        record[(func_name, func_arg_shapes)] = [record[(func_name, func_arg_shapes)][0] + cost_time, record[(func_name, func_arg_shapes)][1] + 1]
        total += cost_time
    record[("total", "")] = (total, 1)
    table = _print_table(record)
    output_csv = perf_json.replace(".json", ".csv")
    with open(output_csv, "w") as f:
        f.write(table.get_csv_string())
    print("================================")
    print("Write table to file: " + output_csv)
    print("================================")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} json_path|perf_file.json")
        exit(1)
    candidate = sys.argv[1]
    if candidate.endswith(".json"):
        _main(candidate)
    elif os.path.isdir(candidate):
        files = os.listdir(candidate)
        json_files = [os.path.join(candidate, f) for f in files if f.endswith(".json")]
        [_main(f) for f in json_files]
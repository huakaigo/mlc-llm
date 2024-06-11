import tvm
from tvm import relay, auto_scheduler
import numpy as np
import onnx
import os
import time
from tvm.contrib import utils
from tvm.contrib import graph_executor
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import autotvm

# 加载 ONNX 模型
log_file = "grounding-dino-tune-726.json"
tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 200,
    "early_stopping": 200,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}
shape_dict = {"input": (1, 3, 800, 1333), "input_ids": (1, 7), "attention_mask": (1, 7), "token_type_ids": (1, 7)}
onnx_model = onnx.load("/home/fuhaiwen1/bigmodel/gd_deploy/convert_onnx/gd_725-sim.onnx", shape_dict)

target = tvm.target.Target("cuda", host="llvm")
dev = tvm.cuda(0)

# 将 ONNX 模型转换为 TVM 的计算图表示
print("Extract tasks...")
relay_module, relay_params = relay.frontend.from_onnx(onnx_model, shape_dict)
# tasks, task_weights = auto_scheduler.extract_tasks(relay_module["main"], relay_params, target)
tasks = autotvm.task.extract_from_program(relay_module["main"], target=target, params=relay_params)

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

tune_tasks(tasks, **tuning_option)

# Compile with the history best
print("Compile...")
with autotvm.apply_history_best(tuning_option["log_filename"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(relay_module, target, params=relay_params)

# save optimized model
path_lib = os.path.join(os.getcwd(), "deploy_lib_tune_726.tar")
lib.export_library(path_lib)
print("save the graph, lib and params into separate files")

# execute the protable graph on TVM
m = graph_executor.GraphModule(lib["default"](dev))
# set inputs
m.set_input("input", tvm.nd.array(np.random.rand(1, 3, 800, 1333).astype(np.float32)))
m.set_input("input_ids", tvm.nd.array(np.random.rand(1, 7).astype(np.int64)))
m.set_input("attention_mask", tvm.nd.array(np.random.rand(1, 7).astype(np.int64)))
m.set_input("token_type_ids", tvm.nd.array(np.random.rand(1, 7).astype(np.int64)))
# execute
for i in range(10):
    start_time = time.time()
    m.run()
    print("%.2f s"%(time.time()-start_time))

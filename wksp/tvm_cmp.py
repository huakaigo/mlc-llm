from tvm.driver import tvmc
model = tvmc.load('resnet50-v2-7.onnx') # 第 1 步：加载
# model.summary() # 打印relay ir
log_file = "tune.log"
print("Start tune ...")
tvmc.tune(model, target="llvm -mcpu=cascadelake", tuning_records=log_file) # 第 1.5 步：可选 Tune
print("End tune ...")

print("Start compile ...")
tvmc.compile(model, target="llvm -mcpu=cascadelake",  tuning_records = log_file, package_path = "./comp", dump_code=["asm", "ll", "tir", "relay"]) # 第 2 步：编译
print("End compile ...")

package = tvmc.TVMCPackage(package_path="./comp")

print("Start run ...")
result = tvmc.run(package, device="cpu") # 第 3 步：运行
print("End run ...")

print(result)
2023-08-28总结
  * 优化方法
    - 融合后算子优化: 搜索ThreadIndex和vectorize参数
    - MLP down算子优化：
      1. reduce维度拆分，提高并行度，降低单线程耗时
      2. 配合ThreadIndex和vectorize参数搜索
  * decode性能优化
    - 单算子
      1. Attention QKV融合kernel优化: 36.448ms  -> 17.824ms,  提升50%
      2. MLP Gate&UP融合Kernel优化: 33.568ms -> 29.344ms, 提升12%
      3. MLP down kernel优化:  33.76ms -> 19.96ms, 提升40%
    - 整网耗时
      ^ 优化前耗时: 127ms
      ^ 应用1,2,3优化: 89ms
  * prefill性能优化
    - 单算子
      1. Attention QKV融合kernel优化: 466.488ms  -> 229.285ms,  提升50%
      2. MLP Gate&UP融合Kernel优化: 544.857ms -> 435.3728ms, 提升20%
      3. MLP down kernel优化:  优化中..
    - 整网耗时
      ^ 优化前耗时： 1602ms
      ^ 应用1.和2.(预计，未实际测试):  1256ms
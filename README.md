# OMP-MPI-CUDA-and-Hybrid-Methods-for-Matrix-Multiplication

[中文](#OMP-MPI-CUDA-and-Hybrid-Methods-for-Matrix-Multiplication-CN)

Parallel and Distributed Computing, High-Performance Computing, GPU-Accelerated Computation, and Matrix Multiplication Experiment

# Versions

1. MPI Version: Microsoft MPI v10.0
2. CUDA Version: 11.7.99
3. Visual Studio Version: 2022

# Compilation

1. CUDA projects and MPI+CUDA projects should be compiled in release mode. The debug mode generates executable files that significantly impact the runtime performance on GPUs.
2. CUDA-related projects include pre-compiled files; however, the runtime environment must have a CUDA version higher than the specified version, i.e., 11.7.99.

# Execution

1. Each subproject is equipped with an auxiliary script with a ".py" suffix for convenient input of program parameters.


# OMP-MPI-CUDA-and-Hybrid-Methods-for-Matrix-Multiplication-CN

 并行计算和分布式计算，高性能计算，GPU显卡计算，矩阵乘法实验

# 版本

1. MPI版本：Microsoft MPI v10.0
2. CUDA版本：11.7.99
3. VisualStudio版本：2022

# 编译

1. CUDA项目和MPI+CUDA项目在编译时，应设置为release编译模式。debug编译模式生成的执行文件对GPU运行速度有较大的影响。
2. CUDA相关的项目保存有编译好的文件，但运行需要CUDA环境版本大于此版本，即11.7.99。

# 运行

1. 每个子项目配备了一个便于输入程序参数的辅助脚本——".py"后缀文件。
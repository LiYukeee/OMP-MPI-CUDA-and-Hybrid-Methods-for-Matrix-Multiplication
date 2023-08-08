# OMP-MPI-CUDA-and-Hybrid-Methods-for-Matrix-Multiplication
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
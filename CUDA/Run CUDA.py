import os
import time
print("Please enter three numbers (M N and K)to determine the order of the matrix")
print("Order of matrix A B and C")
print("A(M×N), B(N×K), C(M×K)")
print("One number represents the multiplication of two square matrices")
print("Separate with spaces: ")

try:
    os.chdir(".//x64//Release")
except:
    try:
        os.chdir(".//x64//Debug")
    except:
        print("Compile first!")
        time.sleep(99999)


while True:
    print("---------------")
    while True:
        try:
            arg = input("input: ")
            arg = arg.strip()
            arg = arg.split(" ")
            if len(arg)==3:
                m = int(arg[0])
                n = int(arg[1])
                k = int(arg[2])
                break
            if len(arg)==1 and len(arg[0])!=0:
                m = n = k = int(arg[0])
                break
            print("error")
            continue
        except:
            print("error")
    os.system(".\\CUDA.exe {} {} {}".format(m, n, k))

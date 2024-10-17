

## Requirements
- CUDA 10.1
- python 3.8.5
- pytorch 1.7.1
- GCC 5.4.0
- cython 0.29.21
- eigency 1.77
- numpy 1.18.1
- torch-geometric 1.6.3 
- tqdm 4.56.0
- ogb 1.2.4
- [eigen 3.3.9] (https://gitlab.com/libeigen/eigen.git)


## Compilation
Cython needs to be compiled before running, run this command:
```
python setup.py build_ext --inplace
```

## Running the code
- On tmall and Patent datasets
```
./run.sh
```


## Installation

Install C++ and Julia (if not already)

```
$ sudo apt-get update
$ sudo apt install build-essential (to get c++ etc.)
```
install Julia (version 1.1.0 recommended if you get errors)
```
$ sudo apt-get install julia
```
Install openblas. Note: you can also use other BLAS libraries like intel MKL, etc.
```
sudo apt-get install libopenblas-dev
```
Clone ITensor and use version 2. Note: need to use version 2!
```
git clone https://github.com/ITensor/ITensor itensor-2
cd itensor-2
git checkout v2
```
Next, follow instructions on https://itensor.org/docs.cgi?page=install for compiling itensor-2 and linking to your BLAS library. 

Before running the example, change the `LIBRARY_DIR` variable in `Makefile` to  point to your itensor2 path. Now you should be able
to run the example 
```
python3 run_example.py
```

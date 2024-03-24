# Temporal consistent mesh decimating 

## Usage

This repository contains the C++ implementation and python binding for a temporal consistent mesh decimating based on libigl. 

Prerequisites:

- Eigen

To install the kernel, first clone this repository recursively:


```bash
git clone --recursive https://github.com/PeizhuoLi/mesh_simplifier.git
```

Then, use pip to install this module in your virtual environment for ganimator:

```bash
conda activate manifold-aware-transformers
pip install ./ganimator-eval-kernel
```

## Acknowledgements

This repository is based on [cmake_example for pybind11](https://github.com/pybind/cmake_example).

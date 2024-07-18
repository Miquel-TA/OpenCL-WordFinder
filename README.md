# OpenCL-WordFinder

This Python program searches for a specified word in a text file using both CPU and GPU processing and compares the performance of these two methods.

## Features

- Generates a random text file if it doesn't exist.
- Uses OpenCL for GPU-accelerated word search.
- Uses Python's `ThreadPoolExecutor` for parallel CPU word search.
- Prints detailed information about the search process and performance.

## Requirements

- Python 3.x
- `numpy`
- `pyopencl`

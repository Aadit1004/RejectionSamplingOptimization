# RejectionSamplingOptimization

This project explores several ways to improve rejection sampling efficiency through better proposal distributions, vectorization, GPU acceleration, and whitening. This project was completed as the final course project for UBC CPSC 440, Advanced Machine Learning, by [Aadit Rao](https://github.com/Aadit1004) and [Matthew Mung](https://github.com/mmung3).

## How to Run

Make sure Python is installed, along with the required libraries:

```
pip install numpy matplotlib scipy "cupy-cuda12x[ctk]"
```

Then move into the `src` directory and run:

```
cd src
python main.py
```
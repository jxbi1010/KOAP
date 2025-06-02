# KOAP
Implementation of ICRA 2025 paper: Imitation Learning with Limited Actions via Diffusion Planners and Deep Koopman Controllers

## Installation
1. Install D3IL benchmark from `https://github.com/ALRhub/d3il` in `environments/d3il`.
2. Follow `environments/d3il/README.md` to register gym environment. 
3. Download dataset to  `environments/dataset/data/`
4. Use `create_small_dataset.py` to generate observation dataset for training.
5. Install Vector-Quantization package for baseline menthods:
```
pip install vector-quantize-pytorch
```

### Reproduce the results
Run training and evaluation with
```
python run_script_<method>.py
```

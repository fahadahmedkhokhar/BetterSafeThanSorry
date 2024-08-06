
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%20-blue.svg)]()

## Requirements

- Python 3.8
- PyTorch

## Installation

The code uses **Python 3.8**.

#### Create a Conda virtual environment:

```bash
conda create --name rcl python=3.8
conda activate rcl
```

#### Clone the project and install requirements:

1. Clone the repository:

```bash
git clone [https://github.com/fahadahmedkhokhar/BetterSafeThanSorry.git]
```
2. Install dependencies:
```bash
pip install -r requirement.txt
```
3. Run Image Classifier:
```bash
python main.py 
```
4. Run Tabular Classifier:
```bash
python tabular_main.py
```
## Compute the Results
1. For Image Classifier:
```bash
python compute_matrix.py
```
2. For Tabular Classifier:
```bash
python tab_compute_matrix.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.


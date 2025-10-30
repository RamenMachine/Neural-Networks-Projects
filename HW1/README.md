# ECE 491 HW1 - Perceptron Implementation

## Project Description
This project implements a perceptron algorithm for binary classification and visualizes a 3D decision boundary plane for ECE 491 coursework.

## Setup Instructions
1. Virtual environment has been created and activated in `venv/`
2. Required packages installed: `numpy`, `matplotlib`

## Running the Code
To run the perceptron homework:
```powershell
.\venv\Scripts\Activate.ps1
python perceptron_hw1.py
```

## Project Structure
- `perceptron_hw1.py` - Main implementation file
- `venv/` - Virtual environment with dependencies
- `.github/copilot-instructions.md` - Project configuration

## Code Features
### Question 1 Part B
- Implements perceptron algorithm for binary classification
- Uses blue and red data points
- Outputs decision boundary equation

### Question 2 Part B  
- Creates 3D visualization of plane: `2x1 + 3x2 + 4x3 - 4 = 0`
- Shows intercept points and triangle visualization
- Interactive 3D plot with matplotlib

## Dependencies
- Python 3.13.7
- NumPy 2.3.2
- Matplotlib 3.10.5

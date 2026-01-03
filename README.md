# MU-based Fatigue Model

This repo contains a Python implementation of a motor-unit (MU) based fatigue / endurance simulation (converted from Potvin & Fuglevand 2017 MATLAB data), with:

- A small Tkinter GUI to enter parameters.
- A matplotlib window that plots simulation outputs and shows a statistics box.
- 
- CSV outputs saved to disk after each run.

## Requirements

- Python 3.x
- Packages:
  - `numpy`
  - `matplotlib`

`tkinter` is included with most standard Python installs on Windows.

## Running

### Option A: Run the GUI (recommended)

This launches a small window where you enter:

- **Percentage of MVIC** (0-100)
- **Duration (seconds)**

Then click **Run Simulation**.

```powershell
python mvic.py
```

The GUI will start a separate process that runs:

```powershell
python mvic.py --run <fthscale> <fthtime>
```

### Option B: Run from the command line (no GUI)

`fthscale` is expressed as a fraction (e.g. `0.80` for 80% MVIC) and `fthtime` is seconds.

```powershell
python mvic.py --run 0.80 20
```

## Outputs

After each run, the script prints endurance-time information to the console.

## Virtual Environment (Windows)

From the project folder:

### 1) Create a virtual environment

```powershell
python -m venv .venv
```

### 2) Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once (in an elevated PowerShell):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
```

### 4) Run

```powershell
python mvic.py
```

Or CLI mode:

```powershell
python mvic.py --run 0.80 20
```

### 5) Deactivate when done

```powershell
deactivate
```

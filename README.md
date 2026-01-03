# Motor Unit Fatigue Simulation

A Python implementation of a motor unit (MU) based fatigue and endurance simulation, converted from Potvin & Fuglevand 2017 MATLAB data. This tool adds visualization of motor unit recruitment, firing patterns, and fatigue during sustained muscle contractions.

## Features

- Interactive Tkinter GUI for parameter input
- Real-time visualization of motor unit activity
- Animated motor unit recruitment and fatigue visualization
- Detailed statistics panel showing:
  - Endurance time
  - Motor unit activation times
  - Recruitment percentage
  - Hypertrophy potential rating
  - Average force output
  - Time to 80% MU recruitment
  - Percentage of MUs exhausted

## Requirements

- Python 3.8+
- Required packages:
  - `numpy`
  - `matplotlib`
  - `tkinter` (usually included with Python on Windows)

## Installation

### Using pip (recommended)

```bash
pip install numpy matplotlib
```

### Option B: Run from the command line (no GUI)

`fthscale` is expressed as a fraction (e.g. `0.80` for 80% MVIC) and `fthtime` is seconds.

```powershell
python mvic.py --run 0.80 20
```

## Outputs

After each run, the script writes CSV files in the project folder, with a prefix matching the contraction intensity (`con`). Example files:

- `0.80 A - Target - Act - Strength (no adapt) - Force - Strength (w adapt).csv`
- `0.80 B - Firing Rate.csv`
- `0.80 C - Individual MU Force Time-History.csv`
- `0.80 D - MU Capacity - relative.csv`

It also prints endurance-time information to the console.

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

2. In the GUI window:
   - Enter the percentage of MVIC (0-100%)
   - Enter the duration in seconds
   - Check "Enable Animation" for real-time visualization
   - Click "Run Simulation"

### Command Line Interface

For automated testing or scripting, you can run the simulation directly:

```bash
# Basic usage
python mvic.py --run <fthscale> <fthtime>

# Example: Run at 80% MVIC for 20 seconds with animation
python mvic.py --run 0.80 20 --animate
```

### Output

The simulation displays several plots:

1. **Panel A**: Excitation and force output over time
2. **Panel B**: Firing rates of motor units over time
3. **Panel C**: Force contributions of motor units over time
4. **Panel D**: MU capacity at endurance time
5. **Statistics Panel**: Detailed performance metrics

## Understanding the Simulation

### Motor Unit Animation
When animation is enabled, you'll see a visualization of motor units:
- Size indicates motor unit size (MU1 smallest to MU120 largest)
- Color intensity shows force contribution (transparent = inactive, deep red = maximum force)
- The animation updates in real-time to show recruitment and fatigue patterns

### Statistics Explained
- **Endurance Time**: Time until force drops below target
- **MU Activation Times**: When each motor unit first becomes active
- **Recruitment %**: Percentage of total motor units recruited
- **Hypertrophy Potential**: Rating based on recruitment percentage
- **Average Force**: Mean force output during the simulation
- **Time to 80% Recruitment**: How long until 80% of MUs are active
- **% MUs Exhausted**: Percentage of motor units that became fatigued

## Troubleshooting

- If the GUI doesn't start, ensure all dependencies are installed
- For animation issues, try updating matplotlib:
  ```bash
  pip install --upgrade matplotlib
  ```
- On Windows, if you get a `ModuleNotFoundError` for tkinter, install it with:
  ```bash
  winget install Python.Python.3.11 --override "ADDLOCAL=ALL"
  ```
  (adjust version number as needed)

## License

This project is for educational and research purposes. Please cite the original work by Potvin & Fuglevand (2017) if used in academic work.

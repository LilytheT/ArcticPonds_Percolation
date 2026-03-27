# Arctic Melt Ponds Percolation Simulator

## Project Overview
This project is a computational physics simulation designed to reproduce the geometric characteristics of Arctic sea ice melt ponds, as described by the **Void Model** in top-tier physics publications. 

By employing a **Poisson Boolean model** with an exponential radius distribution, the simulation drops overlapping circles (representing ice) to form complex, unconnected voids (representing melt ponds). The project investigates continuum percolation phase transitions, emergent fractal geometries, and cluster size power-law distributions.

## Directory Structure
```text
ArcticPonds_Percolation/
├── README.md                  # Project documentation
├── config.nml                 # Simulation configuration file (Namelist)
├── requirements.txt           # Python dependencies
│
├── src/                       # Fortran source code for data generation
│   └── void_model_generator.f90 
│
├── analysis/                  # Python scripts for statistical physics analysis
│   ├── analyze_melt_ponds.py           # Calculates Order Parameter Φ
│   ├── analyze_fractal_dimension.py    # Calculates Fractal Dimension D(A)
│   ├── analyze_size_distribution.py    # Fits Power-law index τ
│   └── visualize_circles.py            # Renders geometry visualizations
│
├── data/                      # Simulation output data (ignored in Git)
│   └── raw/                   # Output .txt coordinate files
│
└── figures/                   # Generated academic plots

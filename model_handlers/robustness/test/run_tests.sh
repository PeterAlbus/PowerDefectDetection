#!/bin/bash


conda activate robustness

# List of attack methods
attack_methods=(
    "FGSM"
    "RFGSM"
    "BIM"
    "PGD"
    "PRGF"
    "SPSA"
    "EAD"
    # "UAP"
    "AutoPGD"
    "BLB"
    "CORRUPT"
    "DEEPFOOL"
    "DIM"
    "Evolutionary"
    "ILLC"
    "LLC"
    "NIM"
    "RLLC"
    "SignHunter"
    "SIGNOPT"
    "SIMBA"
)

# Iterate over each attack method and run the Python script
for method in "${attack_methods[@]}"; do
    python testimport_full.py --attack_method "$method"
done


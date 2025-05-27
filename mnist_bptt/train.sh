#!/bin/bash
# Range of seeds to iterate over
SEED_START=0  # Starting seed value (inclusive)
SEED_END=19   # Ending seed value (inclusive)

# Array of parameters to iterate over
USE_COUPLED_VALUES=(false true)  # Options for use_coupled
HIDDEN_SIZES=(200)               # Hidden sizes

# Loop over each combination of parameters
for use_coupled in "${USE_COUPLED_VALUES[@]}"; do
    for seed in $(seq "$SEED_START" "$SEED_END"); do
        for hid in "${HIDDEN_SIZES[@]}"; do
            JOB_NAME="${use_coupled}-${seed}"
            SLURM_FILE="${JOB_NAME}.slurm"
                
            cat <<EOT > "$SLURM_FILE"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH -c 5
#SBATCH --output=./results1/${JOB_NAME}.log
#SBATCH --error=./errors1/${JOB_NAME}_error.log

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate sage

echo "Running with parameters: use_coupled=${use_coupled}, seed=${seed}, hidden=${hid}"
# Run the Python script with the current parameters
python3 train.py --seed ${seed} --use_coupled ${use_coupled} --hidden ${hid}
EOT

            sbatch "$SLURM_FILE"
            rm "$SLURM_FILE"
        done
    done
done


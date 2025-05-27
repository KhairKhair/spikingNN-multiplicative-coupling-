#!/bin/bash
# Range of seeds to iterate over
SEED_START=0   # Starting seed value (inclusive)
SEED_END=99

# Arrays of parameters to iterate over  
USE_COUPLED_VALUES=(false)        # Options for use_coupled
HIDDEN_VALUES=(30 50 100)          # Options for hidden layer sizes
LAM_VALUES=(0.0)                 # Options for lam
ETA_VALUES=(0.0)                 # Options for eta
BATCH_VALUES=(1 50 100)         # Options for batch size

# Loop over each combination of parameters
for hidden in "${HIDDEN_VALUES[@]}"; do
    for use_coupled in "${USE_COUPLED_VALUES[@]}"; do
        for lam in "${LAM_VALUES[@]}"; do
            for eta in "${ETA_VALUES[@]}"; do
                for batch in "${BATCH_VALUES[@]}"; do
                    # Create directory names based on lam, eta, and batch size
                    LOG_DIR="${lam}-${eta}/logs_${hidden}/batch_${batch}/"
                    ERR_DIR="errors_${lam}-${eta}/batch_${batch}/"
                    mkdir -p "$LOG_DIR" "$ERR_DIR"
                    for seed in $(seq "$SEED_START" "$SEED_END"); do
                        # Create a unique job name using parameter values
                        JOB_NAME="${use_coupled}-${seed}-batch_${batch}"
                        SLURM_FILE="${JOB_NAME}.slurm"
                        # Write the SLURM script for this combination of parameters
                        cat <<EOT > "$SLURM_FILE"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=2:00:00
#SBATCH --mem=6G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}.log
#SBATCH --error=${ERR_DIR}/${JOB_NAME}_error.log
echo "Running with parameters: hidden=${hidden}, use_coupled=${use_coupled}, seed=${seed}, lam=${lam}, eta=${eta}, batch=${batch}"
# Run the Python script with the current parameters
python3 train.py --seed ${seed} --use_coupled ${use_coupled} --hidden ${hidden} --lam ${lam} --eta ${eta} --batch ${batch}
EOT
                        # Submit the SLURM job and remove the SLURM file
                        sbatch "$SLURM_FILE"
                        rm "$SLURM_FILE"
                    done
                done
            done
        done
    done
done


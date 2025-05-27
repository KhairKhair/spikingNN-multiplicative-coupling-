#!/bin/bash
# Range of seeds to iterate over
SEED_START=0
SEED_END=20

# Parameters to iterate over
USE_COUPLED_VALUES=(false true)
SAMPLES_ARRAY=(60000)
HIDDEN_SIZES=(100)

mkdir -p results2 errors2

# Loop over each combination of parameters
for use_coupled in "${USE_COUPLED_VALUES[@]}"; do
    for seed in $(seq "$SEED_START" "$SEED_END"); do
        for samples in "${SAMPLES_ARRAY[@]}"; do
            for hid in "${HIDDEN_SIZES[@]}"; do
                JOB_NAME="${use_coupled}-${seed}"
                SLURM_FILE="${JOB_NAME}.slurm"

                cat <<EOT > "$SLURM_FILE"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --ntasks=1
#SBATCH -c 5  
#SBATCH --time=44:00:00
#SBATCH -o ./results2/${JOB_NAME}.log
#SBATCH -e ./errors2/${JOB_NAME}_error.log


echo "Parameters: use_coupled=${use_coupled}, seed=${seed}, samples=${samples}, hidden=${hid}"

# Run training
python3 train_dcsnn.py --seed ${seed} --use_coupled ${use_coupled} --max_samples ${samples} --hidden_size ${hid}
EOT

                sbatch "$SLURM_FILE"
                rm "$SLURM_FILE"
            done
        done
    done
done


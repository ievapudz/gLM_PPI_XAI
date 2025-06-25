#!/bin/sh

# Part of the automated pipeline to fully train the model after CV for PPI
# prediction

# Requirements:
#   - prepare the parent base.yaml file
#   - generate all representations needed

JOB_PAR_NAME=$1 # parent job name
DEV_SUBSET="train_validate"
DATA_PAR_DIR="./data/"
CONFIGS_PAR_DIR="./configs/"
OUTPUT_PAR_DIR="./outputs/predictions/gLM.models.PredictorPPI/"

declare -A biolm_model
biolm_model["gLM2"]="gLM2_650M"
biolm_model["ESM2"]="esm2_t33_650M_UR50D"
biolm_model["MINT"]="mint"

for representation in "joint_pooling" "separate_pooling" "joint_input_separate_pooling"; do
    for biolm in "gLM2" "ESM2" "MINT"; do
        if [[ "$representation" == joint_input_separate_pooling && "$biolm" == MINT ]]; then
            echo "Skipping $representation for $biolm..."
        else

            # TODO: collect the results from CV: plot the metrics, pick the best epoch of the model
            #       - if any metrics.csv is missing, print out a warning

            # Making directory and the specific job configuration file
            CONFIGS_DIR="${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${representation}/${biolm}/${DEV_SUBSET}/"
            mkdir -p "${CONFIGS_DIR}"
            
            echo "I will write to ${CONFIGS_DIR}/base.yaml..."

            sed "s|JOB_PAR_NAME|${JOB_PAR_NAME}|g" "${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/base.yaml" |\
                sed "s|REPRESENTATION|${representation}|g" | sed "s|BIOLM_MODEL|${biolm_model["${biolm}"]}|g" |\
                sed "s|BIOLM|${biolm}|g" | sed "s|prefix: 'CV'|prefix: ${DEV_SUBSET}|g" | \
                sed "s|emb_dim: 1280|emb_dim: ${emb_dims["$representation"]}|" | \
                sed "s|concat_type: CONCAT|concat_type: ${concat_type["$biolm"]}|" | \
                sed -E "s|^([ \t]+save_dir: .*/)[^/]+/?$|\1"${DEV_SUBSET}"/|" | \
                sed -E "s|^([ \t]+name: .*/)[^/]+/?$|\1"${DEV_SUBSET}"/|" | \
                sed "s|0/metrics|"${DEV_SUBSET}"/metrics|g" | \
                sed "s|0/checkpoints|"${DEV_SUBSET}"/checkpoints|g" | \
                sed "s|kfolds: 5|kfolds: 0|" > "${CONFIGS_DIR}/base.yaml"

            python3 ./gLM/make_train_validate_config.py -c "${CONFIGS_PAR_DIR}" -o "${OUTPUT_PAR_DIR}"\
                -j "${JOB_PAR_NAME}" -r "${representation}" -b "${biolm}" -f 5 --hyperparam "batch_size"
            
            # Run the training and validation
            #bash sh_scripts/sbatch_gpu.sh "${JOB_PAR_NAME}/${representation}/${biolm}/${DEV_SUBSET}" \
            #    "${CONFIGS_DIR}"/base.yaml "" 5
        fi
    done
done

# If dev. subset == test:

# TODO: generate config files for testing (change the run name, add ckpt_path)

# TODO: retrieve testing (point) metrics

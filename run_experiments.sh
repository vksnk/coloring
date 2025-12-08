#!/bin/bash

# 1. DEFINITIONS
# -----------------------------------------------
PYTHON_SCRIPT="train.py"
LOG_FILE="experiment_results.log"

# Define your parameter lists here
# New parameters added below:
CONV_TYPES=("gcn" "gat" "sage" "gin")
NUM_CLASSES_LIST=(8)

# Existing parameters:
HIDDEN_DIMS=(256)
INPUT_DIMS=(16)
NUM_GCNS=(7)

# 2. SETUP
# -----------------------------------------------
# Create or clear the log file with a starting message
echo "Starting Grid Search Experiments at $(date)" > "$LOG_FILE"
echo "-------------------------------------------" >> "$LOG_FILE"

# 3. EXECUTION LOOP
# -----------------------------------------------
# Loop 1: Convolution Type (String)
for c_type in "${CONV_TYPES[@]}"; do

    # Loop 2: Number of Classes (Int)
    for n_class in "${NUM_CLASSES_LIST[@]}"; do

        # Loop 3: Hidden Dimensions
        for h_dim in "${HIDDEN_DIMS[@]}"; do
            
            # Loop 4: Input Dimensions
            for i_dim in "${INPUT_DIMS[@]}"; do
                
                # Loop 5: Number of GCNs
                for n_gcn in "${NUM_GCNS[@]}"; do
                    
                    # Formulate the header string for clarity
                    # Added new flags to the header display
                    HEADER="Running: --conv_type $c_type --num_classes $n_class --hidden_dim $h_dim --input_dim $i_dim --num_of_gcn $n_gcn"
                    
                    # Print header to terminal AND append to log file
                    echo "-------------------------------------------" | tee -a "$LOG_FILE"
                    echo "$HEADER" | tee -a "$LOG_FILE"
                    echo "-------------------------------------------" | tee -a "$LOG_FILE"
                    
                    # Run the python script
                    # Added the new flags to the command execution
                    time python "$PYTHON_SCRIPT" \
                        --conv_type "$c_type" \
                        --num_classes "$n_class" \
                        --hidden_dim "$h_dim" \
                        --input_dim "$i_dim" \
                        --num_of_gcn "$n_gcn" \
                        | tee -a "$LOG_FILE"
                        
                done
            done
        done
    done
done

echo "All experiments finished at $(date)." | tee -a "$LOG_FILE"
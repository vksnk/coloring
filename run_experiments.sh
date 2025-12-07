#!/bin/bash

# 1. DEFINITIONS
# -----------------------------------------------
PYTHON_SCRIPT="train.py" # Replace with your actual filename
LOG_FILE="experiment_results.log"

# Define your parameter lists here
HIDDEN_DIMS=(256)
INPUT_DIMS=(16)
NUM_GCNS=(0 1 2 3 4 5 6 7 8 9)

# 2. SETUP
# -----------------------------------------------
# Create or clear the log file with a starting message
echo "Starting Grid Search Experiments at $(date)" > "$LOG_FILE"
echo "-------------------------------------------" >> "$LOG_FILE"

# 3. EXECUTION LOOP
# -----------------------------------------------
# Loop through Hidden Dimensions
for h_dim in "${HIDDEN_DIMS[@]}"; do
    
    # Loop through Input Dimensions
    for i_dim in "${INPUT_DIMS[@]}"; do
        
        # Loop through Number of GCNs
        for n_gcn in "${NUM_GCNS[@]}"; do
            
            # Formulate the header string for clarity
            HEADER="Running: --hidden_dim $h_dim --input_dim $i_dim --num_of_gcn $n_gcn"
            
            # Print header to terminal AND append to log file
            echo "-------------------------------------------" | tee -a "$LOG_FILE"
            echo "$HEADER" | tee -a "$LOG_FILE"
            echo "-------------------------------------------" | tee -a "$LOG_FILE"
            
            # Run the python script
            # 2>&1 ensures errors are also captured in the log
            time python "$PYTHON_SCRIPT" \
                --hidden_dim "$h_dim" \
                --input_dim "$i_dim" \
                --num_of_gcn "$n_gcn" \
                | tee -a "$LOG_FILE"
                
        done
    done
done

echo "All experiments finished." | tee -a "$LOG_FILE"
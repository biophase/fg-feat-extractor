#!/bin/bash

# Check if the base directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_directory>"
    exit 1
fi

# Base directory for the experiments
base_dir="$1"

# Find all YAML files in the base directory and subdirectories
yaml_files=$(find $base_dir -name "*.yaml")

# Loop through each YAML file found
for yaml_file in $yaml_files; do
    echo "Processing $yaml_file"
    # Construct the script and its arguments
    script_and_args="train.py --config $yaml_file"
    
    # Run the script
    python $script_and_args
    
    # Capture the exit status of the script
    status=$?
    
    # Check if the script exited with an error
    if [ $status -ne 0 ]; then
        echo "Script $script_and_args encountered an error (exit status $status). Moving to the next script."
    fi
done

echo "All scripts have been executed."

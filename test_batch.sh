#!/bin/bash

# Default values for optional arguments
model_version="latest_model.pth"
voxel_size=0.1
return_probs=false
return_embeddings=false

# Usage information
usage() {
    echo "Usage: $0 [-m model_version] [-v voxel_size] [--return_probs] [--return_embeddings] <exp_base_dir>"
    echo ""
    echo "Options:"
    echo "  -m    Model version to test (default: latest_model.pth)"
    echo "  -v    Voxel size for point cloud sampling (default: 0.1)"
    echo "  --return_probs       Include probabilities in the output"
    echo "  --return_embeddings  Include embeddings in the output"
    echo ""
    echo "Arguments:"
    echo "  <exp_base_dir>  Base directory containing experiment subdirectories, e.g., ./exp/batch_1&2"
    exit 1
}

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m) model_version="$2"; shift 2;;
        -v) voxel_size="$2"; shift 2;;
        --return_probs) return_probs=true; shift;;
        --return_embeddings) return_embeddings=true; shift;;
        -h|--help) usage;;
        *) 
            if [ -z "$exp_base_dir" ]; then
                exp_base_dir="$1"
            else
                echo "Unknown argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

# Check if base directory is provided
if [ -z "$exp_base_dir" ]; then
    echo "Error: Base directory for experiments is required."
    usage
fi

# Validate base directory
if [ ! -d "$exp_base_dir" ]; then
    echo "Error: Directory '$exp_base_dir' does not exist."
    exit 1
fi

# Find all experiment directories within the base directory
exp_dirs=$(find "$exp_base_dir" -mindepth 1 -maxdepth 1 -type d)

# Check if any directories are found
if [ -z "$exp_dirs" ]; then
    echo "No experiment directories found in '$exp_base_dir'."
    exit 1
fi

# Loop through each experiment directory and execute the test script
for exp_dir in $exp_dirs; do
    echo "Running tests for experiment directory: $exp_dir"
    
    # Construct the command to execute test.py
    script_and_args="test.py --exp_dir $exp_dir --model_version $model_version --voxel_size $voxel_size"

    # Add optional flags if set
    if [ "$return_probs" = true ]; then
        script_and_args="$script_and_args --return_probs"
    fi
    if [ "$return_embeddings" = true ]; then
        script_and_args="$script_and_args --return_embeddings"
    fi

    # Debugging: print the constructed command
    echo "Executing: python $script_and_args"

    # Execute the test script
    python $script_and_args

    # Capture the exit status of the test script
    status=$?

    # Check if the script exited with an error
    if [ $status -ne 0 ]; then
        echo "Test script encountered an error in directory: $exp_dir (exit status $status). Continuing..."
    fi
done

echo "All test scripts have been executed."

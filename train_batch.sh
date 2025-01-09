# #!/bin/bash

# # Check if the base directory argument is provided
# if [ -z "$1" ]; then
#     echo "Usage: $0 <base_directory>"
#     exit 1
# fi

# # Base directory for the experiments
# base_dir="$1"

# # Find all YAML files in the base directory and subdirectories
# yaml_files=$(find $base_dir -name "*.yaml")

# # Loop through each YAML file found
# for yaml_file in $yaml_files; do
#     echo "Processing $yaml_file"
#     # Construct the script and its arguments
#     script_and_args="train.py --config $yaml_file --wandb_name $wandb_name"
    
#     # Run the script
#     python $script_and_args
    
#     # Capture the exit status of the script
#     status=$?
    
#     # Check if the script exited with an error
#     if [ $status -ne 0 ]; then
#         echo "Script $script_and_args encountered an error (exit status $status). Moving to the next script."
#     fi
# done

# echo "All scripts have been executed."



#!/bin/bash

# Check if the base directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_directory> [--wandb_name <wandb_name>]"
    exit 1
fi

# Base directory for the experiments
base_dir="$1"

# Parse optional --wandb_name argument
wandb_name=""
if [ "$2" == "--wandb_name" ] && [ -n "$3" ]; then
    wandb_name="$3"
fi

# Find all YAML files in the base directory and subdirectories
yaml_files=$(find "$base_dir" -name "*.yaml")

# Loop through each YAML file found
for yaml_file in $yaml_files; do
    echo "Processing $yaml_file"
    
    # Construct the command with or without the --wandb_name argument
    if [ -n "$wandb_name" ]; then
        script_and_args="train.py --config $yaml_file --wandb_name $wandb_name"
    else
        script_and_args="train.py --config $yaml_file"
    fi
    
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

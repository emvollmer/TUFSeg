#!/bin/bash

# Function to handle optional flags
function handle_optional_flag {
    local flag_name="$1"
    local -a values=("${!2}")  # Accept an array of values

    if [[ ${#values[@]} -gt 0 ]]; then
        cmd="$flag_name ${values[@]}"
    else
	cmd=""
    fi
    echo $cmd
}


# Locate the train_UNet.py script relative to the current working directory
train_script_path=$(find . -type f -name "train_UNet.py" -printf '%P\n' | head -n 1)

# Set up the perun command
perun_cmd="perun monitor --format csv -m $train_script_path"


# Initialize empty options and cfg array
src_path=""
dst_path=""
channels_arg=""
processing_arg=""
only_tir=""
cfg_options=()
verbosity=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -src|--src-dir)
            src_path="$2"
            shift 2
            ;;
        -dst|--dst-dir)
            dst_path="$2"
            shift 2
            ;;
        -ch|--channels)
            channels_arg="$2"
            shift 2
            ;;
        -proc|--processing)
            processing_arg="$2"
            shift 2
            ;;
        --only-tir)
            only_tir="$1"
            shift
            ;;
        --cfg-options)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-.* ]]; do
                cfg_options="$cfg_options $1"
                shift
            done
            ;;
        --quiet|-v|-vv)
            verbosity="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Construct the perun command
training_cmd="$perun_cmd -dst $dst_path"

if [ -n "$src_path" ]; then
    training_cmd="$training_cmd -src $src_path"
fi
if [ -n "$channels_arg" ]; then
    training_cmd="$training_cmd --channels $channels_arg"
fi
if [ -n "$processing_arg" ]; then
    training_cmd="$training_cmd --processing $processing_arg"
fi
# Construct the flags for optional arguments
if [ -n "$only_tir" ]; then
    training_cmd="$training_cmd $only_tir"
fi
cfg_options_cmd=$(handle_optional_flag "--cfg-options" cfg_options[@])
training_cmd="$training_cmd $cfg_options_cmd"

if [ -n "$verbosity" ]; then
    training_cmd="$training_cmd $verbosity"
fi

# Run nvidia-smi in the background and redirect its output to gpu_monitoring.log
nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > gpu_monitoring.log &
# Store the process ID (PID) of the background nvidia-smi process
nvidia_smi_pid=$!
stop=0

# Execute the training command
echo "------------------------------------"
echo "Train model by executing command:"
echo "$training_cmd"
echo "------------------------------------"
eval "$training_cmd"

# Check the exit status, if training was Killed or interrupted by keyboard for example
if [ $? -ne 0 ]; then
    echo "Training was terminated with a non-zero exit status."
    stop=1
fi

# Stop the nvidia-smi process
kill "$nvidia_smi_pid"

# Define the model directory using the most recent timestamp folder
model_dir=$(find "$dst_path" -maxdepth 1 -type d -name '2*' -printf '%p\n' | sort | tail -n 1)

# Move the "perun_results" folder into the most recent timestamp directory
if [ -d "perun_results" ]; then
    mv "perun_results" "$model_dir/"
    echo "Moved perun_results folder to $model_dir"
fi
# Move gpu monitoring results into the most recent timestamp directory
if [ -e "gpu_monitoring.log" ]; then
    mv "gpu_monitoring.log" "$model_dir/"
    echo "Moved gpu monitoring log to $model_dir"
fi

# Stop the script run if the process had a non-zero exit status
if [ $stop -eq 1 ] ; then
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

# Get the evaluation module path from the train_script_path by replacing all '/' with '.'
module_path=${train_script_path////.}
eval_module_path=${module_path//train_UNet.py/"evaluate_UNet"}

# Define evaluation command
eval_cmd="python -m $eval_module_path $model_dir"

# Construct the flags for optional arguments
if [ -n "$verbosity" ]; then
    eval_cmd="$eval_cmd $verbosity"
fi

# Execute the evaluation script as a module
echo "------------------------------------"
echo "Evaluate model by executing command:"
echo "$eval_cmd"
echo "------------------------------------"
eval "$eval_cmd"

# Check the exit status, if evaluation was Killed or interrupted by keyboard for example
if [ $? -ne 0 ]; then
    echo "Evaluation was terminated with a non-zero exit status."
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

#!/bin/bash

# Locate the generate_segmentation_masks.py script relative to the current working directory
generate_masks_path=$(find . -type f -name "generate_segm_masks.py" -printf '%P\n' | head -n 1)
generate_masks_path=${generate_masks_path////.}
generate_masks_path=${generate_masks_path/.py/""}

# Predefine optional flags and values so they aren't taken from previous bash script sourcing 
save_for_view=""
dst_path=""
test_size=""
verbosity=""
# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
	-j|--json-dir)
	    json_path="$2"
    	    shift 2
	    ;;
	-i|--img-dir)
	    img_path="$2"
	    shift 2
	    ;;
    	--save-for-view)
	    save_for_view="$1"
	    shift 1
	    ;;
    	-dst)
	    dst_path="$2"
	    shift 2
	    ;;
    	--test-size)
	    test_size="$2"
	    shift 2
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

# Construct the mask generation command with optional flags
generate_masks_cmd="python -m $generate_masks_path -j $json_path -i $img_path $save_for_view $verbosity" 

# Execute the segmentation mask creation command
echo "------------------------------------"
echo "Generate segmentation masks from annotation jsons by executing command:"
echo "$generate_masks_cmd"
echo "------------------------------------"
eval "$generate_masks_cmd"

# Check the exit status, if it interrupted f.e. by keyboard/user
if [ $? -ne 0 ]; then
    echo "Script run was terminated with a non-zero exit status."
    return 1
fi

# Get the test-train-split module path from the generate_segmentation_masks.py
data_split_path=${generate_masks_path//generate_segm_masks/"train_test_split"}

# Construct the data split command with optional flags
data_split_cmd="python -m $data_split_path"
if [ -n "$dst_path" ]; then
    data_split_cmd="$data_split_cmd -dst $dst_path"
fi
if [ -n "$test_size" ]; then
    data_split_cmd="$data_split_cmd --test-size $test_size"
fi
data_split_cmd="$data_split_cmd $verbosity"

# Execute the data split command
echo "------------------------------------"
echo "Define train test dataset split by executing command:"
echo "$data_split_cmd"
echo "------------------------------------"
eval "$data_split_cmd"

# Check the exit status, if it interrupted f.e. by keyboard/user
if [ $? -ne 0 ]; then
    echo "Script run was terminated with a non-zero exit status."
    return 1
fi

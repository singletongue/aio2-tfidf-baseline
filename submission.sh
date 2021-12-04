INPUT_FILE=$1
OUTPUT_FILE=$2

python predict.py \
--model_dir model \
--test_file $INPUT_FILE \
--prediction_file $OUTPUT_FILE

#!/bin/bash
# pip install openai==0.28
# scp -r jones-1:/home/CE/mariak/prompts/prompt/chunked_output/gpt-3.5-turbo/temp0.0/de_min_srclos_vratio2 prompt/chunked_output/gpt-3.5-turbo/temp0.0
# chmod +x prompting/feeding_chunks.sh
# print each command and its arguments to the standard error output before executing them
# bash -x prompt/feeding_chunks.sh lazy 0.7
# sh prompt/feeding_chunks.sh min_triad_vratio2 0.7
# sh prompt/feeding_chunks.sh lazy 0.7 reworks

# The chunks which timed_out need to be copied over to a prompt/_reworks/de_lazy/ or prompt/_reworks/en_lazy/ and run again from this bash script
# see ids of timed out chunks in logs: prompt/logs/feeding/
# edit the input_folder!

for my_lang in "de" "en"; do
  echo "${my_lang}"
  #my_lang="en"

  # for  in "lazy"#  "min_srclos_vratio2" "min_tgtlos_vratio2" "min_triad_vratio2" "tiny2_triad_vratio3" "seg_expert"; do
  my_mode="$1"

  # Extract the float value from the command-line argument
  temperature="$2"

  level="$3"

  gpt="$4"

  mode="$5"

  input_folder=""
  outto=""

  # Reworks input
  if [ "$mode" = "reworks" ]; then
    input_folder="4_prompting/_reworks/input/${gpt}/temp${temperature}/${my_lang}_${level}_${my_mode}/"
    outto="4_prompting/_reworks/output/"  # output folders are made and formated in the py script
    echo "Reworks mode ON"
  else  # assume empty string
    # Specify the input folder chunked_input/${my_mode}  or testing (for timeout chunks)
    input_folder="4_prompting/chunked_input/${my_lang}_${level}_${my_mode}/"
    outto="4_prompting/chunked_output/"
    echo "Running initial re-writing"
  fi

  echo "input: $input_folder"
  echo "outto: $outto"

  logdir="logs/feeding"
  mkdir -p ${logdir}

  log_file="${logdir}/${my_mode}_feeding_error_t-${temperature}.log"

  # Remove the existing log file if present
  rm -f "$log_file"

  # Function to log messages to both terminal and file
  log_message() {
      echo "$1" | tee -a "$log_file"
  }

  # Iterate over files in the input folder
  for file in "$input_folder"*; do
      # Check if the file is a regular file (not a directory)
      if [ -f "$file" ]; then
          log_message "Processing file: $file"

          # Call your Python script with arguments
          python3 4_prompting/api_prompting.py --lang "$my_lang" --tempr "$temperature" --res "$outto" --chunk_path "$file" --model "$gpt" 2>&1 | tee -a "$log_file"
      fi
  done

  echo "DONE for $my_lang $temperature $my_mode!"

done

echo "Both languages for $temperature $my_mode! have run"
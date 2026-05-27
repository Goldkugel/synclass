#!/bin/bash

export PYTHONUSERBASE="/work/${USER}/.local"
source /opt/miniconda3/bin/activate nlp_global

GPUS="0,1"

#MODE="test" 
MODE=""

#CoT="chain-of-thoughts" 
CoT=""
#FS="" 
FS="few-shot"

DEFINITION="definition"
COMMENT="comment"
CHILDREN="children"
PARENTS="parents"

MODELS=(
  #"mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "google/medgemma-4b-it"
  #"Qwen/Qwen3-4B-Instruct-2507"
  #"mistralai/Mistral-7B-Instruct-v0.2"
  #"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  #"google/medgemma-27b-text-it"
)

clear

[ -f "../data/output/transform/transform.csv" ] || (python3 "transform.py" && python3 "embed.py" "" "" "test" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS" && python3 "embed.py" "" "" "" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS")

for MODEL in "${MODELS[@]}"; do
  #echo $MODEL
  #sbatch classify.sh "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "" "" "" ""
  #sbatch classify.sh "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
  sbatch classify.sh "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
  #sbatch classify.sh "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
  sbatch classify.sh "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
done

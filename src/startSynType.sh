GPUS="7,2"

MODE="test" 
#MODE=""

CoT="chain-of-thoughts" 
#CoT=""
FS="" 
#FS="few-shot"

MODELS=(
  "google/medgemma-27b-text-it"
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "Qwen/Qwen3-4B-Instruct-2507"
  "google/medgemma-4b-it"
)

clear

./prepare.sh "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" 

for MODEL in "${MODELS[@]}"; do
  python3 ./syntype.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS"
done

for MODEL in "${MODELS[@]}"; do
  python3 ./syntypeformat.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS"
done

python3 "./syntypemerge.py" "" "" "$MODE" "$CoT" "$FS"
python3 "./syntypeeval.py" "" "" "$MODE" "$CoT" "$FS"

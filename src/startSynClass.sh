GPUS="6,7"

MODE="test" # =""

CoT="" # "chain-of-thoughts" #=""
FS="few-shot" #=""

# python3 ./synclass.py "google/medgemma-27b-text-it" "0,1"

MODELS=(
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "google/medgemma-4b-it"
  "Qwen/Qwen3-4B-Instruct-2507"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "google/medgemma-27b-text-it"
)

clear

./prepare.sh "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS"

for MODEL in "${MODELS[@]}"; do
  python3 ./synclass.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS"
done

for MODEL in "${MODELS[@]}"; do
  python3 ./synclassformat.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS"
done

python3 "./synclassmerge.py" "" "" "$MODE" "$CoT" "$FS"
python3 "./synclasseval.py" "" "" "$MODE" "$CoT" "$FS"

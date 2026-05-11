GPUS="4,7"

MODE="test" 
#MODE=""

CoT="chain-of-thoughts" 
#CoT=""
FS="" 
#FS="few-shot"

DEFINITION="definition"
#DEFINITION=""
COMMENT="comment"
#COMMENT=""
CHILDREN="children"
#CHILDREN=""
PARENTS="parents"
#PARENTS=""

MODELS=(
  #"mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  #"google/medgemma-4b-it"
  #"Qwen/Qwen3-4B-Instruct-2507"
  #"mistralai/Mistral-7B-Instruct-v0.2"
  #"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "google/medgemma-27b-text-it"
)

clear

[ -f "../data/output/transform/transform.csv" ] || (python3 "transform.py" && python3 "embed.py" "" "" "test" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS" && python3 "embed.py" "" "" "" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS")

for MODEL in "${MODELS[@]}"; do
  python3 ./synclass.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "" "" "" ""
  python3 ./synclass.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
  #python3 ./synclass.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
  python3 ./synclass.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
  python3 ./synclass.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
done

for MODEL in "${MODELS[@]}"; do
  python3 ./synclassformat.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "" "" "" ""
  python3 ./synclassformat.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
  python3 ./synclassformat.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
  python3 ./synclassformat.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
  python3 ./synclassformat.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
done

python3 ./synclassmerge.py "" "" "$MODE" "$CoT" "$FS" "" "" "" ""
python3 ./synclassmerge.py "" "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
python3 ./synclassmerge.py "" "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
python3 ./synclassmerge.py "" "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
python3 ./synclassmerge.py "" "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""

python3 ./synclasseval.py "" "" "$MODE" "$CoT" "$FS" "" "" "" ""
python3 ./synclasseval.py "" "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
python3 ./synclasseval.py "" "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
python3 ./synclasseval.py "" "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
python3 ./synclasseval.py "" "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""

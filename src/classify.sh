#!/bin/bash
#SBATCH --job-name=synclass
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: HH:MM:SS)
#SBATCH --output=../data/logs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=../data/logs/job_%j.err         # Speicherort für Fehlermeldungen

GPUS="$2"

MODEL="$1"
MODE="$3"
CoT="$4"
FS="$5"

DEFINITION="$6"
COMMENT="$7"
CHILDREN="$8"
PARENTS="$9"

python3 ./synclass.py "$MODEL" "$GPUS" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
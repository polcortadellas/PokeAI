#!/bin/bash

set -e

RELEASES_DIR="releases"
mkdir -p "$RELEASES_DIR"

RUN_NUM=1
while [ -d "$RELEASES_DIR/run_$RUN_NUM" ]; do
    RUN_NUM=$((RUN_NUM+1))
done

CURRENT_RELEASE="$RELEASES_DIR/run_$RUN_NUM"
mkdir -p "$CURRENT_RELEASE"

echo "=========================================="
echo "🚀 PIPELINE: CREATING VERSION $RUN_NUM"
echo "=========================================="

echo -e "\n🧠 [1/3] Starting training of 5M steps..."
.venv/bin/python main.py --mode train --total-timesteps 5000000

echo -e "\n🔍 [2/3] Searching for 'best_model.zip'..."

BEST_MODEL="checkpoints/best_model/best_model.zip"

if [ ! -f "$BEST_MODEL" ]; then
    echo "❌ ERROR: best_model.zip not found. Did you configure EvalCallback in main.py?"
    exit 1
fi

echo -e "✅ Best model found. Copying to the vault..."
cp "$BEST_MODEL" "$CURRENT_RELEASE/champion_model.zip"

REPORT_FILE="$CURRENT_RELEASE/eval_report.md"

echo -e "\n📊 [3/3] Evaluating the champion and generating .md..."

echo "# 🏆 Evaluation Report - Version $RUN_NUM" > "$REPORT_FILE"
echo "**Date:** $(date)" >> "$REPORT_FILE"
echo "**Model:** \`champion_model.zip\`" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "## 📈 Audit Results" >> "$REPORT_FILE"

echo '```text' >> "$REPORT_FILE"

.venv/bin/python main.py --mode eval --model "$CURRENT_RELEASE/champion_model.zip" --headless --steps 15000 >> "$REPORT_FILE" 2>&1

echo '```' >> "$REPORT_FILE"

echo -e "\n🎉 Pipeline completed! Version $RUN_NUM has been saved safely in $CURRENT_RELEASE."
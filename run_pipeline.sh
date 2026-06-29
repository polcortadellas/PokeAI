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
echo "🚀 PIPELINE: CREANDO VERSIÓN $RUN_NUM"
echo "=========================================="

echo -e "\n🧠 [1/3] Iniciando entrenamiento de 1M de pasos..."
.venv/bin/python main.py --mode train --total-timesteps 5000000

echo -e "\n🔍 [2/3] Buscando el 'best_model.zip'..."

BEST_MODEL="checkpoints/best_model/best_model.zip"

if [ ! -f "$BEST_MODEL" ]; then
    echo "❌ ERROR: No se encontró el best_model.zip. ¿Configuraste el EvalCallback en main.py?"
    exit 1
fi

echo -e "✅ Mejor modelo encontrado. Copiando a la caja fuerte..."
cp "$BEST_MODEL" "$CURRENT_RELEASE/champion_model.zip"

REPORT_FILE="$CURRENT_RELEASE/reporte_eval.md"

echo -e "\n📊 [3/3] Evaluando al campeón y generando .md..."

echo "# 🏆 Reporte de Evaluación - Versión $RUN_NUM" > "$REPORT_FILE"
echo "**Fecha:** $(date)" >> "$REPORT_FILE"
echo "**Modelo:** \`champion_model.zip\`" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "## 📈 Resultados de la Auditoría" >> "$REPORT_FILE"
echo '```text' >> "$REPORT_FILE"

.venv/bin/python main.py --mode eval --model "$CURRENT_RELEASE/champion_model.zip" --headless --steps 15000 >> "$REPORT_FILE" 2>&1

echo '```' >> "$REPORT_FILE"

echo -e "\n🎉 ¡Pipeline completado! La Versión $RUN_NUM ha sido guardada a salvo en $CURRENT_RELEASE."
#!/bin/bash
# Custom test script for PyTorch-only DLRM with pre-trained model

# Ruta a tu modelo preentrenado
MODEL_PATH="./dlrm_model.pth"

# Check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi

dlrm_py="python ../dlrm_s_pytorch.py"

echo "🔧 Ejecutando tests SOLO en PyTorch con modelo preentrenado..."

# Ejecutar tests PyTorch con datos aleatorios y tu modelo
echo $dlrm_py
$dlrm_py --mini-batch-size=1 --data-size=1 --nepochs=1 --arch-interaction-op=dot \
         --arch-sparse-feature-size=8 --arch-mlp-bot="4-4-8" --arch-mlp-top="64-32-1" \
         --learning-rate=0.1 --debug-mode --load-model="$MODEL_PATH" --inference-only $dlrm_extra_option > ppp1

$dlrm_py --mini-batch-size=2 --data-size=4 --nepochs=1 --arch-interaction-op=dot \
         --arch-sparse-feature-size=8 --arch-mlp-bot="4-4-8" --arch-mlp-top="64-32-1" \
         --learning-rate=0.1 --debug-mode --load-model="$MODEL_PATH" --inference-only $dlrm_extra_option > ppp2

$dlrm_py --mini-batch-size=2 --data-size=5 --nepochs=1 --arch-interaction-op=dot \
         --arch-sparse-feature-size=8 --arch-mlp-bot="4-4-8" --arch-mlp-top="64-32-1" \
         --learning-rate=0.1 --debug-mode --load-model="$MODEL_PATH" --inference-only $dlrm_extra_option > ppp3

$dlrm_py --mini-batch-size=2 --data-size=5 --nepochs=3 --arch-interaction-op=dot \
         --arch-sparse-feature-size=8 --arch-mlp-bot="4-4-8" --arch-mlp-top="64-32-1" \
         --learning-rate=0.1 --debug-mode --load-model="$MODEL_PATH" --inference-only $dlrm_extra_option > ppp4

echo "✅ Tests ejecutados solo en PyTorch con el modelo cargado."

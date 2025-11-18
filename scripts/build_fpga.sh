#!/bin/bash
cd $(dirname "$0")/..

python3 -m tensorflow.python.tools.freeze_graph \
  --input_saved_model_dir=models/ensemble.h5 \
  --output_graph=../quantization/quant_out/frozen.pb \
  --output_node_names="softmax"

vai_q_tensorflow \
  --input_frozen_graph ../quantization/quant_out/frozen.pb \
  --input_fn ../quantization/calib_dataset.py \
  --output_dir ../quantization/quant_out \
  --method 1

vai_c_tensorflow \
  --frozen_pb ../quantization/quant_out/quantized.pb \
  --arch ../arch/KV260.json \
  --output_dir ../compilation/compiled \
  --net_name emg_cnn

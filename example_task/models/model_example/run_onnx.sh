MODEL_PATH=/user_home/du.jing/国家安全/单标签/model/20260612_165846/best_checkpoint
python3 convert_to_onnx.py \
--model_dir  ${MODEL_PATH} \
--seq_max_length 512 \
--num_heads 12 \
--hidden_size 312 \
--device 'cpu' \
--onnx_file ${MODEL_PATH}/model.onnx \
--opt_onnx_file  ${MODEL_PATH}/model_opt.onnx

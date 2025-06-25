#!/bin/bash
#./inference.sh 11-16-23-44-13-E3-w-acc-best javascript 10 0  /home/liwei/Code-Watermark/variable-watermark/results/embed_extract/11-16-23-44-13-E3-w-acc-best-embed-repete-logit-0.jsonl  /home/liwei/Code-Watermark/variable-watermark/results/embed_extract/11-16-23-44-13-E3-w-acc-best-extracted-repete-logit-0.jsonl
set -e  # 任何命令失败时立即退出
python embedder.py --model_name $1 --lang $2 --topk $3 --device $4 -o $5
python extractor.py --model_name $1 --lang $2 --device $4 -i $5 -o $6
python /home/liwei/Code-Watermark/variable-watermark/ybr/pots_jsonl_eval.py --jsonl_path $6 --lang $2
python /home/liwei/Code-Watermark/variable-watermark/ybr/jsonl_codebleu_eval.py  $2  $6
python /home/liwei/Code-Watermark/variable-watermark/var_sim.py --lang $2 --filepath $6
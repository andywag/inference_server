#!/bin/bash
export GM2=mk2
#export POPLAR_LOG_LEVEL=INFO
#export POPART_LOG_LEVEL=INFO
# export POPLAR_ENGINE_OPTIONS='{"autoReport.directory":"/localdata/xianw/profiles/gpt2/small_model_fast","debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true"}'
#128
# python tasks/text_generator_fast.py \
#       --model gpt2 \
#       --fp16 true \
#       --single-ipu true \
#       --input-len  64 \
#       --output-len 64 \
# # #512
# python tasks/text_generator_fast.py \
#       --model-path uer/gpt2-chinese-poem \
#       --fp16 true \
#       --single-ipu true \
#       --input-len  256 \
#       --output-len 256 \
# #1024
# python tasks/text_generator_fast.py \
#       --model-path uer/gpt2-chinese-poem \
#       --fp16 true \
#       --single-ipu true \
#       --poptorch_loop true \
#       --input-len  512 \
#       --output-len 512 \
# #1024
# python tasks/text_generator_fast.py \
#       --model gpt2 \
#       --fp16 true \
#       --single-ipu true \
#       --input-len  512 \
#       --output-len 512 \

# # #xl 128
# python tasks/text_generator_fast.py \
#       --model gpt2-xl \
#       --fp16 true \
#       --layers-per-ipu 0 6 7 7 7 7 7 7 \
#       --input-len  50 \
#       --output-len 14 \

#1 token small using loop
#python text_generator_fast_one_token.py \
#      --model-path gpt2-medium \
#      --fp16 true \
#      --single-ipu true \
#      --poptorch_loop true \
#      --batch-size 1 \
#      --input-len  100 \
#      --output-len 924 \

##1 token small
python text_generator_fast_one_token.py \
      --model-path gpt2 \
      --fp16 true \
      --single-ipu true \
      --poptorch_loop false \
      --batch-size 1 \
      --input-len  16 \
      --output-len 768 \

##1 token small
# python tasks/text_generator_fast_one_token.py \
#       --model gpt2 \
#       --fp16 true \
#       --single-ipu true \
#       --input-len  64 \
#       --output-len 64 \


# 1 token xl
# python tasks/text_generator_fast_one_token.py \
#       --model gpt2-xl \
#       --fp16 true \
#       --layers-per-ipu 0 6 7 7 7 7 7 7 \
#       --input-len  50 \
#       --output-len 14 \
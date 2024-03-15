lm_eval --model hf --model_args pretrained=$1,dtype="float"  --tasks triviaqa  --device cuda:0 --output_path triviaqa

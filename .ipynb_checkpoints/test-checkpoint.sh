file_name="pipeline-gpt2-50-90-w0.3-b4-aug-cl-zero-shot"
python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-initial-0-r/ --input_file=data/eval_topics.jsonl.0 --output_file=outputs/${file_name}-0-0.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-0-1-r/ --input_file=data/eval_topics.jsonl.0 --output_file=outputs/${file_name}-0-1.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-0-2-r/ --input_file=data/eval_topics.jsonl.0 --output_file=outputs/${file_name}-0-2.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-0-3-r/ --input_file=data/eval_topics.jsonl.0 --output_file=outputs/${file_name}-0-3.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-initial-1-r/ --input_file=data/eval_topics.jsonl.1 --output_file=outputs/${file_name}-1-0.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-1-1-r/ --input_file=data/eval_topics.jsonl.1 --output_file=outputs/${file_name}-1-1.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-1-2-r/ --input_file=data/eval_topics.jsonl.1 --output_file=outputs/${file_name}-1-2.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-1-3-r/ --input_file=data/eval_topics.jsonl.1 --output_file=outputs/${file_name}-1-3.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-initial-2-r/ --input_file=data/eval_topics.jsonl.2 --output_file=outputs/${file_name}-2-0.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-2-1-r/ --input_file=data/eval_topics.jsonl.2 --output_file=outputs/${file_name}-2-1.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-2-2-r/ --input_file=data/eval_topics.jsonl.2 --output_file=outputs/${file_name}-2-2.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-2-3-r/ --input_file=data/eval_topics.jsonl.2 --output_file=outputs/${file_name}-2-3.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-initial-3-r/ --input_file=data/eval_topics.jsonl.3 --output_file=outputs/${file_name}-3-0.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-3-1-r/ --input_file=data/eval_topics.jsonl.3 --output_file=outputs/${file_name}-3-1.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-3-2-r/ --input_file=data/eval_topics.jsonl.3 --output_file=outputs/${file_name}-3-2.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-3-3-r/ --input_file=data/eval_topics.jsonl.3 --output_file=outputs/${file_name}-3-3.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-initial-4-r/ --input_file=data/eval_topics.jsonl.4 --output_file=outputs/${file_name}-4-0.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-4-1-r/ --input_file=data/eval_topics.jsonl.4 --output_file=outputs/${file_name}-4-1.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-4-2-r/ --input_file=data/eval_topics.jsonl.4 --output_file=outputs/${file_name}-4-2.jsonl

python cqr/run_prediction.py --model_path=pipeline_models/${file_name}-4-3-r/ --input_file=data/eval_topics.jsonl.4 --output_file=outputs/${file_name}-4-3.jsonl

cd outputs
cat ${file_name}-0-0.jsonl ${file_name}-1-0.jsonl ${file_name}-2-0.jsonl ${file_name}-3-0.jsonl ${file_name}-4-0.jsonl > ${file_name}-0.jsonl

cat ${file_name}-0-1.jsonl ${file_name}-1-1.jsonl ${file_name}-2-1.jsonl ${file_name}-3-1.jsonl ${file_name}-4-1.jsonl > ${file_name}-1.jsonl

cat ${file_name}-0-2.jsonl ${file_name}-1-2.jsonl ${file_name}-2-2.jsonl ${file_name}-3-2.jsonl ${file_name}-4-2.jsonl > ${file_name}-2.jsonl

cat ${file_name}-0-3.jsonl ${file_name}-1-3.jsonl ${file_name}-2-3.jsonl ${file_name}-3-3.jsonl ${file_name}-4-3.jsonl > ${file_name}-3.jsonl
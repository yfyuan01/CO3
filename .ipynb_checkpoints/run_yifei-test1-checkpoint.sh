python cqr/pipeline.py --train_gold_file data/eval_topics.jsonl --train_unlabeled_file_s data/unlabeled_data/unlabeled_self_learn.jsonl --train_unlabeled_file_r data/unlabeled_data/unlabeled_canard.jsonl --output_dir pipeline_models/pipeline-gpt2-20-60-w0.3 --aug_data_weight 0.3 --model_name_or_path=gpt2 --per_gpu_train_batch_size=4 --save_steps=-1 --cross_validate --confidence_threshold_s 20 --confidence_threshold_r 60 

python cqr/pipeline.py --train_gold_file data/eval_topics.jsonl --train_unlabeled_file_s data/unlabeled_data/unlabeled_self_learn.jsonl --train_unlabeled_file_r data/unlabeled_data/unlabeled_canard.jsonl --output_dir pipeline_models/pipeline-gpt2-30-80-w0.2 --aug_data_weight 0.3 --model_name_or_path=gpt2 --per_gpu_train_batch_size=4 --save_steps=-1 --cross_validate --confidence_threshold_s 30 --confidence_threshold_r 80 

python cqr/pipeline.py --train_gold_file data/eval_topics.jsonl --train_unlabeled_file_s data/unlabeled_data/unlabeled_self_learn.jsonl --train_unlabeled_file_r data/unlabeled_data/unlabeled_canard.jsonl --output_dir pipeline_models/pipeline-gpt2-30-90-w0.2 --aug_data_weight 0.3 --model_name_or_path=gpt2 --per_gpu_train_batch_size=4 --save_steps=-1 --cross_validate --confidence_threshold_s 30 --confidence_threshold_r 90 
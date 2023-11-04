# Zero-shot samples
python hlm/app/evaluate_qa.py --dataset MATH --dataset-args="dict(levels=5, subsample=20, include_asy=False)" --dataset-name=math_hard_20 --reasoning-args="dict(prompt='0shot.txt', prompt_tag='0shot', temperature=0.7, n=64)"

# Question retrieval
python hlm/app/evaluate_qa.py --dataset MATH --dataset-args="dict(levels=5, subsample=20, include_asy=False)" --dataset-name=math_hard_20 --reasoning-fn explore --reasoning-args="dict(prompt='dynamic_1shot.txt', type='retrieval-question', retrieval_ratio={'question': 1}, n_retrievals=4, temperature=0.7, n=32, demo_kwargs=dict(dataset_name='MATH', include_asy=False, split='train'))"

# High-level hints and tactics
python hlm/app/evaluate_qa.py --dataset MATH --dataset-args="dict(levels=5, subsample=20, include_asy=False, aug='/data/Projects/AGI/projects/exploration/results/gpt4_answer_techniques.json')" --dataset-name=math_hard_20 --reasoning-fn explore --reasoning-args="dict(prompt='use_techniques.txt', type='key-techniques', n=16, demo_kwargs=dict(dataset_name='MATH', include_asy=False))"

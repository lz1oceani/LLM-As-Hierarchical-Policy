# Unleashing the Creative Mind: Language Model As Hierarchical Policy For Improved Exploration on Challenging Problem Solving 

This repo contains the code, prompts, and model outputs for [Unleashing the Creative Mind: Language Model As Hierarchical Policy For Improved Exploration on Challenging Problem Solving](https://arxiv.org/pdf/2311.00694.pdf).

Current methods for enabling Large Language Models (LLMs) to solve challenging reasoning problems often fall short due to limited exploration capabilities. In this work, we enhance LLMs' problem-solving strategies by framing them as a **hierarchical policy** through in-context learning. This consists of a visionary **leader** proposing diverse high-level tactics and a **follower** executing detailed processes per each tactic. We introduce a tournament-based approach to efficiently select the best solution from the generated solution groups. Our approach fosters strategic exploration, generates insightful hints, and improves answer accuracy on challenging problems in the MATH dataset.

Here are teasers for our methods:
![method](https://github.com/lz1oceani/LLM-As-Hierarchical-Policy-Test/blob/master/images/teaser.jpg)
![example](https://github.com/lz1oceani/LLM-As-Hierarchical-Policy-Test/blob/master/images/example.jpg)


## Setup
You need to first have an OpenAI API key and store it as the environment variable ``OPENAI_API_KEY`` (see [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)).

Installation: ``pip install -e .``

## Experiments
In the [folder of prompts](./hlm/prompts/), we have provided the prompts we used in this to instruct LLMs to generate .

If you would like to perform 


## File Structure and Data Format
``results/math_hard_20`` contains human annotations for the 6 tasks we experimented with. For each task, we sample 50 valid reasoning chains and 50 reasoning chains exhibiting mistakes. It has the following format:

```javascript
{
  "question", // question
  "answer", // Natural Program reasoning chain output (to be verified)
  "final_answer", // ground truth solution of this question
  "correct", // final answer correctness
  "flag": 1, // label given by annotator; flag=1 means the reasoning chain is valid; flag=0 means the reasoning chain has mistakes
}
```

## Citation
Please cite our paper if you find our idea helpful. Thanks a lot!

```
@article{ling2023unleashing,
  title={Unleashing the Creative Mind: Language Model As Hierarchical Policy For Improved Exploration on Challenging Problem Solving},
  author={Ling, Zhan and Fang, Yunhao and Li, Xuanlin and Mu, Tongzhou and Lee, Mingu and Pourreza, Reza and Memisevic, Roland and Su, Hao},
  journal={arXiv preprint arXiv:2311.00694},
  year={2023}
}
```

## License

This project is licensed under the CC-BY-2.0 License.

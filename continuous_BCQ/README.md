# Off-Policy Deep Reinforcement Learning without Exploration

Code for Batch-Constrained deep Q-Learning (BCQ). If you use our code please cite the [paper](https://arxiv.org/abs/1812.02900).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.4](https://github.com/pytorch/pytorch) and Python 3.6. 

### Overview

If you are interested in reproducing some of the results from the paper, a behavioral policy (DDPG) needs to be trained by running:
```
python main.py --train_behavioral --gaussian_std 0.1
```
This will save the PyTorch model. A new buffer, corresponding to the "imperfect demonstrations" task, can then be collected by running:
```
python main.py --generate_buffer --max_timesteps 100000
```
Or for the "imitation" task by running:
```
python main.py --generate_buffer --gaussian_std 0.0 --rand_action_p 0.0
```
Finally train BCQ by running:
```
python main.py
```

Settings can be adjusted with different arguments to main.py.

DDPG was updated to learn more consistently. Additionally, with version updates to Python, PyTorch and environments, results may not correspond exactly to the paper. Some people have reported instability using the v2 environments, so sticking with v3 is preferred.

### Bibtex

```
@inproceedings{fujimoto2019off,
  title={Off-Policy Deep Reinforcement Learning without Exploration},
  author={Fujimoto, Scott and Meger, David and Precup, Doina},
  booktitle={International Conference on Machine Learning},
  pages={2052--2062},
  year={2019}
}
```

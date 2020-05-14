# Off-Policy Deep Reinforcement Learning without Exploration

Code for Batch-Constrained deep Q-Learning (BCQ). If you use our code please cite the [paper](https://arxiv.org/abs/1812.02900).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.4](https://github.com/pytorch/pytorch) and Python 3.6. 

### Overview

If you are interested in reproducing some of the results from the paper, a behavioral policy (DDPG) needs to be trained by running 
```
main.py --train_behavioral
```
This will save the PyTorch model. A new buffer can then be collected by running
```
main.py --generate_buffer
```
Finally train BCQ by running
```
main.py
```

Settings can be adjusted with different arguments to main.py. 

DDPG was updated to learn more consistently. Additionally, with version updates to Python, PyTorch and environments, results may not correspond exactly to the paper. 

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
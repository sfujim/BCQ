# Discrete Batch-Constrained deep Q-Learning (BCQ)

Code for Batch-Constrained deep Q-Learning (BCQ) for discrete actions. Additionally, we include a full pipeline for training DQN in Atari. If you use our code please cite our [Deep RL workshop NeurIPS 2019 paper](https://arxiv.org/abs/1910.01708).

Repo is setup for Atari and toy tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.4](https://github.com/pytorch/pytorch) and Python 3.6. 

### Overview

To begin a behavioral policy (DQN) needs to be trained by running 
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

Settings can be adjusted by changing the dicts in main.py. This is a reproduction of the code in the original paper, and results will not correspond exactly.

### Bibtex

```
@article{fujimoto2019benchmarking,
  title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
  author={Fujimoto, Scott and Conti, Edoardo and Ghavamzadeh, Mohammad and Pineau, Joelle},
  journal={arXiv preprint arXiv:1910.01708},
  year={2019}
}
```
# Off-Policy Deep Reinforcement Learning without Exploration

Code corresponding to the paper. If you use our code please cite the [paper](https://arxiv.org/abs/1812.02900).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 0.4](https://github.com/pytorch/pytorch) and Python 2.7. 

### Overview

Main algorithm, Batch-Constrained Q-learning (BCQ), can be found at BCQ.py.

If you are interested in reproducing some of the results from the paper, an expert policy (DDPG) needs to be trained by running train_expert.py. This will save the expert model. A new buffer can then be collected by running generate_buffer.py and adjusting the settings in the code or using the default settings. 

If you are interested in the standard forward RL tasks with DDPG or TD3, check out my other [Github](https://github.com/sfujim/TD3). 

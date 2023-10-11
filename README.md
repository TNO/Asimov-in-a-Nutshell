<p align="center">
	<img src="https://github.com/ckitsanelis/Nutshell_Test/blob/45f19859dc8c324d0b5cbc9833b3081d068d372e/asimov-logo.png" width="500px"/>
</p>

[Website](https://asimov-project.eu) | [Publications](https://itea4.org/project/asimov.html)

The ASIMOV project (AI training using Simulated Instruments for Machine Optimization and Verification) is a use-case driven, collaborative research project running under the ITEA4 programme. The consortium consists of 11 partners across two nations (Netherlands and Germany), which include large industrial parties, leading universities, research institutes, and SME’s.

# "ASIMOV in a Nutshell" demonstrator

The ASIMOV-in-a-Nutshell demonstrator was developed to illustrate the ASIMOV project vision in a simple setting, by taking inspiration from the electron microscopy use case. Furthermore, this demonstrator can be utilised to explore the impact of Digital Twin modelling decisions and the systems engineering challenges imposed by Reinforcement Learning. For fast exploration in terms of both development time and runtime without high hardware requirements, the demonstrator ignores detailed physics modeling and case-specific image recognition as much as possible.

The demonstrator optimizes a noisifier system that produces images with two types of noise, gaussian noise and blur, which were chosen to imitate the types of aberrations produced by the electron microscope. The system produces noise images based on a collection of non-noisy base images, and two knobs that control the applied noise levels. At the beginning of each episode a base image is selected randomly, and afterwards the system can be operated by repeatedly observing the produced noisy image and by adjusting both knobs. The ultimate goal is to develop an RL agent that operates the system knobs until an image with minimal noise is obtained. 

## Experimental Setup

The experimental set-up for the ASIMOV-in-a-Nutshell demonstrator is summarized in the below figure. We use rectangles to represent processing steps and ovals to represent data artifacts. The processing steps are grouped using dashed lines to indicate modules that conform to the ASIMOV reference architecture.

<p align="center">
	<img src="https://github.com/ckitsanelis/Nutshell_Test/blob/b23c77eccc3ce20cb291a6029ce0a614ad4486bb/demonstrator-setup.png" width="500px"/>
</p>

The blue elements are directly related to the image noisifier. Its output is a noisy image, and its inputs are the collection of base images and the applied noise levels. When all applied noise levels are set to 0, the noisy image is identical to the base image.
The orange elements are directly related to the Reinforcement Learning technology that we intend to use for the optimizer agent. Its outputs are knob actions and its inputs are the observation and reward. The reward is only used during the training phase; the other inputs and outputs are used both in the training and operational phase. Finally, the green elements represent the intermediate components that link the blue and orange elements:
- The noise classifier is a separate supervised learning AI component that takes a noisy image and estimates the applied noise levels.
- The knob distortion computes the applied noise levels based on the actual knob positions. To ensure that the agent does not know the optimal knob positions in advance, the knob distortion adds linear knob offsets that are selected randomly at the beginning of each episode.
- The knob action interpreter computes the actual knob positions based on knob actions. To experiment with different knob sensitivities between multiple similar systems (for example within a product family, or between a real system and a digital twin), it is also possible to apply linear knob multipliers that are selected randomly at the beginning of each episode.
- The reward computation is only used during the training phase, and it accesses the real applied noise level. This information would not be externally visible on a real implementation of the noisifier system, but it is accessible in a DT of the noisifier system.

## Quick Start Guide

### Environment Creation

The easiest way to get the "ASIMOV in a Nutshell" demonstrator running is by constructing an Anaconda environment and installing all necessary dependencies using pip. You can opt to run the demonstrator in your own environment, in which case you may also need to upgrade pip to successfully install all of the dependencies: `pip install --upgrade pip`

```bash
# create a conda environment with the specified name and python version 3.10.
conda create -n <ENV_NAME> python=3.10

# activate the created environment
conda activate <ENV_NAME>

# install the necessary libraries with pip utilising the requirements.txt file in the repo. 
pip install -r requirements.txt
```

### Data Collection

In this repository, you may notice that both of the data subdirectories (one to train and one to test) have been kept empty. In this way, we encourage users to populate this directory with their own dataset for their trials. Once both of the subdirectories have been populated, users need only to specify the path to the dataset, the desired image size and colormode in the `config.yaml` file.

### Multi-classifiers based on Supervised Learning (SL)

In the first step, users have to train a noise classifier that takes an image and estimates the applied noise levels. This step can be achieved using the file `1_supervised_learning.py`. Running this file with the argument `--generate` will read the images stored in the training subdirectory, produce variations of each with different levels of noise applied and store them in the directory structure expected by the keras data loader. Moving forward, the generated images can be utilised to train a model by utilising using the argument `--train`. This repository contains multiple neural network architectures which can be used for this (including LeNet-5, VGG16 and Xception), however, users may also opt to configure their own architectures. The `config.yaml` file should be utilised to specify which architecture to utilise, the path to store the produced models and to tweak certain model hyperparameters.

An example of the commands that must be run before moving on to the next step is shown below:
```bash
# Train a model to detect the amount of Gaussian Noise in an image:
python 1_supervised_learning.py --generate --gaussian
python 1_supervised_learning.py --train --gaussian

# Train a model to detect the amount of Blur in an image:
python 1_supervised_learning.py --generate --blur
python 1_supervised_learning.py --train --blur
```

### Independent optimizer training based on Reinforcement Learning (RL)

In the following step, users have to train a reinforcement learning agent that interacts with knob settings to minimise the noise levels in an image. For this purpose, we have built a custom made environment named `env_asimov_nutshell_1d_noise`, which follows the Gymnasium standard API and is designed to be lightweight, fast, and easily customizable. An RL agent can be trained on the environment by running the file `2_reinforcement_learning_1d.py` with the argument `--train`. The agent retrieves the estimated noise levels and actual knob positions from the environment as observation input. In each step, rewards can thus be calculated by computing whether an action lead to a favorable adjustement of the knob. As the default algorithm for Reinforcement Learning we use Soft Actor-Critic, however once again, users may also opt to implement other strategies. After an agent has been trained, users can see them perform in real time using the argument `--apply`. The `config.yaml` file should be utilised to specify which RL algorithm to utilise, the path to store the produced models and to tweak training hyperparameters.

An example of the commands to be run is shown below:
```bash
# Train a reinforcement learning agent to minimise the amount of Gaussian Noise in an image:
python 2_reinforcement_learning_1d.py --train --gaussian

# Train a reinforcement learning agent to minimise the amount of Blur in an image:
python 2_reinforcement_learning_1d.py --train --blur
```

### Combined optimizer application based on Reinforcement Learning (RL)

Finally, users may observe how both of the agents perform simultaneously by running the following command:

```bash
python 3_reinforcement_learning_2d.py --apply
```

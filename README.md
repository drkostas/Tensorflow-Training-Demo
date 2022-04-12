# COSC525: Project 3: Train with Tensorflow

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/COSC525-Project3/blob/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [Installing the requirements](#installing)
  + [Using the Makefile](#installing_makefile)
  + [Manual Installations](#installing_manually)
+ [Running the code](#run_locally)
    + [Execution Options](#execution_options)
        + [main.py](#src_main)
+ [Todo](#todo)
+ [License](#license)

## About <a name = "about"></a>

Project 3 for the Deep Learning course (COSC 525). Training various networks with Tensorflow.

The main code is located in the [main.py](main.py) file. The Neuron, FullyConnectedLayer, 
and NeuralNetwork classes are located in the [src folder](src).

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python = 3.10 and any Bash based shell (e.g. zsh) installed.

```ShellSession
$ python3.10 -V
Python 3.10

$ echo $SHELL
/usr/bin/zsh
```

## Installing the requirements <a name = "installing"></a>

### Using the Makefile <a name = "installing_makefile"></a>
All the installation steps are being handled by the [Makefile](Makefile).

Then, to create a conda environment, install the requirements, setup the library and run the tests
execute the following commands:

```ShellSession
$ make create_env
$ conda activate cosc525_project3
$ make requirements
```

### Manual Installations <a name = "installing_manually"></a>

For manual installation, you can create a virtual environment 
and install the requirements by executing the following commands:

```ShellSession
$ conda create -n cosc525_project3 -y python=3.10
$ conda activate cosc525_project3
$ conda install --file requirements.txt -y
```

## Running the code <a name = "run_locally"></a>

### Execution Options <a name = "execution_options"></a>

First, make sure you are in the correct virtual environment:

```ShellSession
$ conda activate cosc525_project3

$ which python
/home/<user>/anaconda3/envs/src/bin/python
```

#### main.py <a name = "src_main"></a>

In order to run the code use the --help option for instructions:

```ShellSession
    $ python train.py --help
```

```ShellSession
    $ python evaluate.py --help
```

## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

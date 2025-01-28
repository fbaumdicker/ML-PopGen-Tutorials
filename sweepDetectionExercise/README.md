# Sweep detection + machine learning exercise

The instructions for this exercise are found in the sweeps.pdf file. So you can just open that up and work your way through it when you are ready. But first, there are some things you will need to install:

# Preparation for sweep detection exercise

All of the programs to be used during this exercise run on a Unix, so if your laptop runs Linux or mac OS you should be able to skip to the second section and begin installing the software directly. Windows users, on the other hand, need to make some adjustments before installing the other packages.

Throughout this document, every line that starts with the `>` (greater than) character needs to be entered into the command line of your terminal.

These instructions were mostly written by Matheus Salles, with a few edits from me (and help from Google Translate).

## 1) WINDOWS USERS ONLY: install WSL (https://learn.microsoft.com/en-us/windows/wsl/install)
This is not strictly necessary for the workshop, as you can just get a Google Colab account and then run commands in the Colab notebook by prefacing them with the `!` character (otherwise it will think you are typing python code), or you can run a terminal from Colab by following some of the advice here: https://stackoverflow.com/questions/59318692/how-can-i-run-shell-terminal-in-google-colab. It is actually pretty quick and easy. I especially like the `colab-xterm` tool that someone shared in the replies.

Still, it may be useful to have a Linux environment on your Windows machine for running things locally, so I recommend trying this out if you have time:

You can install everything you need to run WSL with just one command. Open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting "Run as administrator." Then, enter the following command:
> wsl --install 

This command enables the features needed to run WSL and install the Ubuntu Linux distribution. After the installation finishes, restart your computer.

## 2) Installing necessary software
All workshop activities can be carried out directly on the terminal for mac OS and Linux users and on the WSL terminal for Windows users. Now let's move on to installing the programs.


### 2a) Install miniconda
Although it is not necessary, it is highly recommended that you create a virtual computing environment for the workshop. To do that, you will need a python package installer like miniconda (). If you already have one then you can skip this step.

#### Linux and WSL users:
> mkdir -p ~/miniconda3
> 
> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
> 
> bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
> 
> rm -rf ~/miniconda3/miniconda.sh

#### mac users:
> mkdir -p ~/miniconda3
> 
> curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
>
> bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
>
> rm -rf ~/miniconda3/miniconda.sh

### 2b) Install `diploSHIC`, a machine learning method for sweep detection.

####First, create a virtual environment with Python 3.9, and name it `diploshic`
> conda create -n diploshic python=3.9 --yes

#### Now activate the virtual environment and install diploSHIC:
> conda activate diploshic #use this command whenever you need to run diploshic
> 
> pip install diploshic
> 
> diploSHIC -h #verify that installation was successful


#### In some cases, installing diploSHIC via pip ends up generating an installation error. If this is your case, first remove diploSHIC using the following commands:
> conda env remove --name diploshic
> 
> pip cache purge

#### Then restart the terminal and then enter the following commands
> conda create -n diploshic python=3.9 --yes
> 
> conda activate diploshic
> 
> sudo apt-get install gfortran
> 
> conda install numpy
> 
> pip install --no-cache-dir diploshic

#### Use the following command to verify that your install was successful
> diploshic -h

#### Whenever you are done using diploSHIC, you can deactivating the virtual environment you created for it as follows:
> conda deactivate

### 2b) Install the coalescent simulator `discoal`
> git clone https://github.com/kr-colab/discoal.git
> 
> cd discoal
> 
> make discoal

#### Use the following command to verify that your discoal install was successful
> ./discoal

# Sweep detection + machine learning exercise

The instructions for this exercise are found in the sweeps.pdf file. So you can just open that up and work your way through it when you are ready. But first, there are some things you will need to install:

# Preparation for sweep detection exercise

All of the programs to be used during this exercise run on Unix, so if your laptop runs Linux you can skip to the third section and begin installing the software directly. This used to be the case for Mac OS as well, but unfortunately some recent changes to some of the packages used to build one of our software tools has complicated things considerably for Mac Users who will need to go to section 2--this may take some time so I apologize in advance. Windows users also need to make some adjustments before installing the other packages, described in section 1 below.

Throughout this document, every line that starts with the `$` (greater than) character needs to be entered into the command line of your terminal.

These instructions were mostly written by Matheus Salles, with a few edits from me (and help from Google Translate).

## 1) WINDOWS USERS ONLY: install WSL (https://learn.microsoft.com/en-us/windows/wsl/install)
You can install everything you need to run WSL with just one command. Open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting "Run as administrator." Then, enter the following command:
> $ wsl --install 

This command enables the features needed to run WSL and install the Ubuntu Linux distribution. After the installation finishes, restart your computer.

## 2) Installing necessary software
All workshop activities can be carried out directly on the terminal for mac OS and Linux users and on the WSL terminal for Windows users. Now let's move on to installing the programs.


### 2a) Install miniconda
Although it is not necessary, it is highly recommended that you create a virtual computing environment for the workshop. To do that, you will need a python package installer like miniconda (https://docs.anaconda.com/miniconda/). If you already have one then you can skip this step.

#### Linux and WSL users:
> $ mkdir -p ~/miniconda3
> 
> $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
> 
> $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
> 
> $ rm -rf ~/miniconda3/miniconda.sh

#### Mac users:
> $ mkdir -p ~/miniconda3
> 
> $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
>
> $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
>
> $ rm -rf ~/miniconda3/miniconda.sh

### 2b) Install `diploSHIC`, a machine learning method for sweep detection.

#### First, create a virtual environment with Python 3.9, and name it `diploshic`
> $ conda create -n diploshic python=3.9 --yes

#### Now activate the virtual environment and install diploSHIC:
> $ conda activate diploshic #use this command whenever you need to run diploshic

After this step, Mac Users should skip the steps below and go straight to section 2c.

#### Install diploSHIC:
> $ pip install diploshic
> 
> $ diploSHIC -h #verify that installation was successful

#### In some cases, installing diploSHIC via pip ends up generating an installation error. If this is your case, first remove diploSHIC using the following commands:
> $ conda env remove --name diploshic
> 
> $ pip cache purge

#### Then restart the terminal and then enter the following commands
> $ conda create -n diploshic python=3.9 --yes
> 
> $ conda activate diploshic
> 
> $ sudo apt-get install gfortran
> 
> $ conda install numpy
> 
> $ pip install --no-cache-dir diploshic

### 2c) Mac users: battle with the compiler and other build settings to get diploSHIC working.
Okay, this is the annoying part. I won't elaborate on these commands too much. Just run them and everything should work out in the end, but it took quite some time on my machine (~30 minutes):

> #install homebrew if you haven't already
> $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
>
> #install the necessary compilers
> $ brew install gcc gfortran libomp
> 
> #clone the diploSHIC github repo
> $ git clone https://github.com/kern-lab/diploSHIC.git
>
> #navigate to the diploSHIC repo
> $ cd diploSHIC

At this point you will have to open `pyproject.toml` in the diploSHIC repo and edit the line that begins with `requires` to be:

`requires = ["setuptools<60.0.0", "numpy"]`

Then save that change, close the `pyproject.toml` file and then continue on:

> #set some environment variables
>  $ export SYSTEM_VERSION_COMPAT=1
>
> $ export LDFLAGS="-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"
>
> #finally, build the diploSHIC package
> $ pip install .

### 2d) Use the following command to verify that your install was successful
> $ diploshic -h

#### Whenever you are done using diploSHIC, you can deactivating the virtual environment you created for it as follows:
> $ conda deactivate

### 2e) Install the coalescent simulator `discoal`
> $ git clone https://github.com/kr-colab/discoal.git
> 
> $ cd discoal
> 
> $ make discoal

#### Use the following command to verify that your discoal install was successful
> ./discoal

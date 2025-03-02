# PoMoBalance exercises for Cold Spring Habor pre-workshop for ProbGen 2025 Meeting

## Workshop

The ProbGen meeting includes a workshop  that will run on the first day pre-meeting. The workshop consists of talks and demo's including practical exercises for PoMoBalance based on RevBayes tutorial.

The PoMoBalance tutorial can be found at: 
* [PoMoBalance](https://revbayes.github.io/tutorials/pomobalance/)

### Software

Before the workshop starts, please install the following software on your own computer:

* RevBayes. For this workshop, we will be using the ```development``` RevBayes branch, where the polymorphism-aware models are all available. Make sure you follow the instructions below to install RevBayes.
Other helpful software: 
* [Tracer](https://beast.community/tracer)
* [FigTree](https://github.com/rambaut/figtree/releases)
* [R](https://www.r-project.org/) or [RStudio](https://rstudio.com/) with the package **Rcpp**
* A good text editor, such as Sublime Text, NotePad++, Atom, BBEdit, vim, or emacs. Word is not a good text editor!

## Install RevBayes

### Compile on Linux

Follow the installation steps [here](https://revbayes.github.io/compile-linux). 

### Compile on Mac OS X

Follow the installation steps [here](https://revbayes.github.io/compile-osx). 

### Compile on Windows

Follow the installation steps [here](https://revbayes.github.io/compile-windows). However, we will not be using the standard revbayes branch for this workshop, but the one that has the PoMo models. Clone the ```dev_PoMo``` repository instead of the master one. This is done by substituting the following step  

```
 git clone https://github.com/revbayes/revbayes.git revbayes
```

by this one:

```
git clone --branch development https://github.com/revbayes/revbayes.git revbayes
git check out 
```


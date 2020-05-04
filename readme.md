# CUDA PCISPH

Predictive-Corrective Incompressible Smooth Particle Hydrodynamics developed for NVIDIA CUDA. It is a tool for solving Navier-Stokes' equations in a particle-based fluid simulation called Smooth Particle Hydrodynamics. This method predicts and corrects future positions for each particle forcing the fluid's imcompressibility.

## Getting Started

**NOTE**: This project can only be run on **Windows OS**.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Your machine needs to be **CUDA capable**, i.e., it needs to have a CUDA-enabled Graphics Card Unit (GPU). For more info about CUDA-enabled GPUs, please refer to [NVIDIA official developer website](https://developer.nvidia.com/cuda-gpus) or the [CUDA page on Wikipedia](https://en.wikipedia.org/wiki/CUDA).

Before proceeding with the CUDA installation, you should first intall the latest version of [Visual Studio](https://visualstudio.microsoft.com/) with **Desktop Development with C++*. For detailed info, refer to [this tutorial](https://www.youtube.com/watch?v=IsAoIqnNia4).

After the installation of Visual Studio has been succesfully completed, if you have a CUDA-capable GPU, you must install the latest version of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Select your OS and version accordingly, it does not matter if you choose _exe(local)_ or _exe(network)_. For detailed info and how to test installation, refer to [this tutorial](https://www.youtube.com/watch?v=cL05xtTocmY).

After the installation of CUDA Toolkit, open Windows Command Prompt and type ```nvcc --version```. You should get the following answer:

```Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:32:27_Pacific_Daylight_Time_2019
Cuda compilation tools, release 10.2, V10.2.89```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Visual Studio](https://visualstudio.microsoft.com/) - the latest version of Visual Studio with **C++ Desktop Development Tool**.

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

This project was developed by José Antonio Rico Coque and Bartosz Powalka in [West Pomeranian University of Technology](https://www.zut.edu.pl/uczelnia/aktualnosci.html) (ZUT) on the [Faculty of Mechanical Engineering and Mechatronics](https://wimim.zut.edu.pl/index.php?id=11909).


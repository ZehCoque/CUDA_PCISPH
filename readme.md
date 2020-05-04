# CUDA PCISPH

Predictive-Corrective Incompressible Smooth Particle Hydrodynamics developed with NVIDIA CUDA for GPU multithreading. It is a tool for solving Navier-Stokes' equations in a particle-based fluid simulation called Smooth Particle Hydrodynamics in much faster than applications run only with CPUs. This method predicts and corrects future positions for each particle forcing the fluid's incompressibility.

## Getting Started

**NOTE**: This project can only be run on **Windows OS**.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Your machine needs to be **CUDA capable**, i.e., it needs to have a CUDA-enabled Graphics Card Unit (GPU). For more info about CUDA-enabled GPUs, please refer to [NVIDIA official developer website](https://developer.nvidia.com/cuda-gpus) or the [CUDA page on Wikipedia](https://en.wikipedia.org/wiki/CUDA).

Before proceeding with the CUDA installation, you should first install the latest version of [Visual Studio](https://visualstudio.microsoft.com/) with **Desktop Development with C++**. For detailed info, refer to [this tutorial](https://www.youtube.com/watch?v=IsAoIqnNia4).

After the successful installation of Visual Studio, if you have a CUDA-capable GPU, you must install the latest version of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Select your OS and version accordingly, it does not matter if you choose _exe(local)_ or _exe(network)_. For detailed info and how to test the installation, please refer to [this tutorial](https://www.youtube.com/watch?v=cL05xtTocmY).

After the installation of CUDA Toolkit, open Windows Command Prompt and type ```nvcc --version```. You should get something similar to:

```
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:32:27_Pacific_Daylight_Time_2019
Cuda compilation tools, release 10.2, V10.2.89
```

## Physical properties and initial conditions

The physical properties and initial conditions should be set in the [/props](https://github.com/ZehCoque/CUDA_PCISPH/tree/master/props) directory. Each file is self-explanatory. This is where you can change the simulation for your desired conditions.

## Deployment

To deploy this project, you should build it with Visual Studio. Open ```CUDA_PCISPH.sln``` in Visual Studio, select ```Release x64``` in the solution configuration just left of the green _play button_ on the top of the screen and go to the **Solution Explorer** (keyboard shortcut ```Ctrl+Alt+L```). Right click on ```CUDA_PCISPH``` and select the ```Build``` option. After it's done, go to the same folder of ```CUDA_PCISPH.sln``` and look for the ```\bin``` directory. Open it and go to the ```\x64``` folder and look for the ```.exe``` file. This is the application. You can move this file to another directory. **Just remember** to move the ```\props``` directory as well. Double click on it to run.

## Post processing

All the simulation results will be on the ```/results``` directory, created in the same directory the ```.exe``` is located. The post processing can be done using [Paraview](https://www.paraview.org/). It can read the files that end with ```.pvd``` and ```.vtu``` generated by this project. The ```.pvd``` file consists in a group of files with time markings and a ```.vtu``` is a single instance io time of the simulation. Open ```PCISPH.pvd``` and ```boundary.vtu``` to start post processing the simulation.

## Contributing

Please read [contributing.md](https://github.com/ZehCoque/CUDA_PCISPH/blob/master/contributing.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

This project was developed by ***Jos� Antonio Rico Coque*** and ***Bartosz Powalka*** in [West Pomeranian University of Technology](https://www.zut.edu.pl/uczelnia/aktualnosci.html) (ZUT) on the [Faculty of Mechanical Engineering and Mechatronics](https://wimim.zut.edu.pl/index.php?id=11909).

## References

[1](https://www.zora.uzh.ch/id/eprint/29726/1/pcisph.pdf) Solenthaler, B. & Pajarola, R. Predictive-Corrective Incompressible SPH. (2009)

[2](https://www.researchgate.net/publication/221622694_Boundary_Handling_and_Adaptive_Time-stepping_for_PCISPH) Ihmsen, M., Akinci, N., Gissler, M. & Teschner, M. Boundary handling and adaptive time-stepping for PCISPH. 

[3](https://people.inf.ethz.ch/~sobarbar/papers/Sol12/Sol12.pdf) Akinci, N., Ihmsen, M., Akinci, G., Solenthaler, B. & Teschner, M. Versatile Rigid-Fluid Coupling for Incompressible SPH. 

[4](https://github.com/openworm/sibernetic) Sibernetic




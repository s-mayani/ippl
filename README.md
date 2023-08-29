# Independent Parallel Particle Layer (IPPL)
Independent Parallel Particle Layer (IPPL) is a performance portable C++ library for Particle-Mesh methods. IPPL makes use of Kokkos (https://github.com/kokkos/kokkos), HeFFTe (https://github.com/icl-utk-edu/heffte), and MPI (Message Passing Interface) to deliver a portable, massively parallel toolkit for particle-mesh methods. IPPL supports simulations in one to six dimensions, mixed precision, and asynchronous execution in different execution spaces (e.g. CPUs and GPUs). 

## Installing IPPL and its dependencies

### Requirements
The following libraries are required:

* MPI
* [Kokkos](https://github.com/kokkos) >= 4.1.00
* [HeFFTe](https://github.com/icl-utk-edu/heffte) >= 2.2.0; only required if IPPL is built with FFTs enabled (`ENABLE_FFT=ON`)

To build IPPL and its dependencies, we recommend using the [IPPL build scripts](https://github.com/IPPL-framework/ippl-build-scripts). See the [documentation](https://github.com/IPPL-framework/ippl-build-scripts#readme) for more info on how to use the IPPL build script.

## IPPL Tutorial and Presentations
A brief introduction to IPPL and its functionalities can be found in this [tutorial](https://github.com/IPPL-framework/ippl-presentations/blob/master/ippl_tutorial/ippl_introduction_slides.pdf)

You can find other presentations based on IPPL [here](https://github.com/IPPL-framework/ippl-presentations) 


## Contributions
We are open and welcome contributions from others. Please open an issue and a corresponding pull request in the main repository if it is a bug fix or a minor change.

For larger projects we recommend to fork the main repository and then submit a pull request from it. Please follow the coding guidelines as mentioned in this [page](https://github.com/IPPL-framework/ippl/blob/master/WORKFLOW.md) 

You can add an upstream to be able to get all the latest changes from the master. For example, if you are working with a fork of the main repository, you can add the upstream by:
```bash
$ git remote add upstream git@github.com:IPPL-framework/ippl.git
```
You can then easily pull by typing
```bash
$ git pull upstream master
````
All the contributions (except for bug fixes) need to be accompanied with a unit test. For more information on unit tests in IPPL please
take a look at this [page](https://github.com/IPPL-framework/ippl/blob/master/UNIT_TESTS.md).

## Citing IPPL

```
@article{muralikrishnan2022scaling,
  title={Scaling and performance portability of the particle-in-cell scheme for plasma physics applications
         through mini-apps targeting exascale architectures},
  author={Muralikrishnan, Sriramkrishnan and Frey, Matthias and Vinciguerra, Alessandro and Ligotino, Michael and
          Cerfon, Antoine J and Stoyanov, Miroslav and Gayatri, Rahulkumar and Adelmann, Andreas},
  journal={arXiv preprint arXiv:2205.11052},
  year={2022}
}
```

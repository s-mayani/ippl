This is the branch corresponding to the paper "Error Analysis and Parallel Scaling Study of A Parareal Parallel-in-Time Integration Algorithm for Particle-in-Fourier Schemes". The installation 
needs a host compiler (e.g. gcc), GPU-aware MPI installation and CMake. The dependencies are (with the versions used for running the simulations in the brackets)
1) Kokkos (4.2.00)
2) Heffte (2.4.0)
3) FINUFFT (2.2.0 specifically commit 871bb8fe)
4) IPPL

The installation script from the repository https://github.com/srikrrish/IPPL_install_scripts/blob/main/ippl_A100_gpu_parapif.sh installs all the dependencies
and compiles the mini-apps for the JUWELS booster suercomputer with A100 GPUs. For other systems/GPUs necessary tweaks may be needed. 

Once compiled the space-only parallel versions of the mini-apps can be found in `ippl/build_with_kokkos_4_2_00_heffte_2_4_0_finufft_2_2_0_psmpi_2024/alpine/ElectrostaticPIF` 
whereas the space-time parallel versions using the parareal algorithm are in `ippl/build_with_kokkos_4_2_00_heffte_2_4_0_finufft_2_2_0_psmpi_2024/alpine/PinT`.

The data and the post-processing scripts can be found in the repository https://github.com/srikrrish/parapif-figures/tree/parapif-paper-v1.0.0

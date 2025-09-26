# hand-sign-translation-kv260
*Demo for CPS Summer School 2025*:

This demo showcases an application that recognizes hand sign language and converts it into text in real time. The system is deployed on the AMD Kria KV260 platform using the PYNQ framework. The classification task is performed by a CNN, which is accelerated on the FPGA’s Programmable Logic (PL) via the DPU, ensuring the low-latency requirements of the service are met.


## How to train, optimize and compile the model for KV260:
Pull the docker image of Vitis AI for model optimization and compiling for the specific target:
```
docker pull xilinx/vitis-ai-cpu:2.5
```
Run the docker from the correct directory:
```
cd <your_path>/training
./docker_run.sh xilinx/vitis-ai-cpu:2.5
```
Activate the pytorch env and run the all-in-1 script:
```
conda activate vitis-ai-pytorch
source run_all.sh
```

NOTE: the compiled model will be stored in the build directory, then you have to move it in the KV260 (e.g. via `scp`).

## Red Team:
- @smanoni (*UNIBO*)
- @creinwar (*ETH*)
- @balditommaso (*SSSA*)
- @martinowiczzz (*UP*)

# APAI25-LAB07-PULP-Tiling-Part1

**Assignment DEADLINE: 02/01/2026 (at 15:30)**

## Material

Assignment: [here](docs/assignment.docx)
Slides: [here](docs/slides.pdf)
Board setup: [here](docs/board-setup.pdf)

## How to deliver the assignment:

Use Virtuale, upload only the assignment file named as follows:

`LAB#_APAI_name.pdf`


# Part1: Deploying a CNN on GAP9 with NNTool 

This first tutorial is under the [Part1-NNTool](./Part1-NNTool/) folder. This task is done only in LAB1, on the GAP9 boards. 
To setup the board (ONLY IN LAB1!!), follow the instructions in [board-setup](./docs/board-setup.pdf).

## Run the CNN on GAP9

First, connect the board to the PC and setup GAP-SDK following the setup guide.

To quantize the model, generate and compile its code and run it on GAP9 with:
- python nntool_generate_model.py


# Part2: Deploying a Transformer model on a generic platform with Deeploy

This second tutorial is under the [Part2-Deeploy](./Part2-Deeploy/) folder. This setup requires your PC and the WSL2 machine you used for the previous labs.

## Setting up your environment

This setup requires a different Docker image with respect to the one used in the previous labs. To this aim:
1. Open your WSL2 machine (Ubuntu24) that you already installed for the previous labs.
2. Clone and enter this repository with:
```
git clone https://github.com/EEESlab/APAI25-LAB10-End-to-End-Deployment.git
cd APAI25-LAB10-End-to-End-Deployment/
```
3. Open this folder in VSCode with:
```
code .
```
4. As the folder opens, click on "Re-open in container" to install and run the Docker image reserved to this lab. 
5. Set up Deeploy with: 
```
git config --global --add safe.directory /workspaces/APAI25-LAB10-End-to-End-Deployment/Part2-Deeploy
git submodule update --init --recursive
cd Part2-Deeploy && pip install -e . && cd ..
```
6. Enjoy the assignment :)


## Doing the assignment and running the code

To this end, please follow the instructions in the [assignment](docs/assignment.pdf).

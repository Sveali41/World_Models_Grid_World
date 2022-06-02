# Deep Learning and Applied Ai Project: 
## Dennis Rotondi, World Models in MiniGrid

### How to reproduce the results:

- clone this repository on your machine
```sh
git clone https://github.com/DennisRotondi/dlai_project.git
```
- install the requirements inside the cloned folder
```sh
pip3 install -r requirements.txt
```
- install the package
```sh
python3 setup.py install
```
- create the dataset
```sh
python3 src/env/grid_save.py 
```
- train the models {vae, mdnrnn, controller} (pick one each time, in this order)
```sh
python3 src/models/{vae, mdnrnn, controller}_train
```
- - or request access to [my dvc remote folder](https://drive.google.com/drive/u/1/folders/1fXQrD-7vmVolooEzNEojThRN_beHLOu2) (dvc on google drive requires it, also if the folder were public it still need to be shared individually) and pull my uploaded files for dataset and ckpt
```sh
dvc pull
```
- now make sure to have the visualize option set to true in conf/hparams/config.yaml and enjoy watching the agent solving the game... or at least trying
```sh
python3 src/env/play.py
```

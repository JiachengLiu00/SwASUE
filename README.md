# SwASUE



## Installation
The code runs with recent Pytorch versions, e.g. 1.4. 
Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/), the most important packages can be installed as:
```shell
conda create --name SwASUE python=3.7
conda activate SwASUE
conda install pytorch torchvision torchaudio cudatoolkit=<CUDA_VERSION> -c pytorch
conda install matplotlib scipy scikit-learn pandas
conda install -c conda-forge faiss-gpu
pip install pyyaml easydict termcolor tqdm simplejson yacs
```
We refer to the `requirements.txt` file for an overview of the packages in the environment we used to produce our results.

## Training

### Setup
The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/cifar10`.
- Specify the output directory in `configs/env.yml`. All results will be stored under this directory. 

Our experimental evaluation includes the following datasets: CIFAR10, CIFAR100-20, STL10 and ImageNet. The ImageNet dataset should be downloaded separately and saved to the path described in `utils/mypath.py`. Other datasets will be downloaded automatically and saved to the correct path when missing.

### Train model
First, you need Start with a pre-training task. The configuration files can be found in the `configs/` directory. The training procedure consists of the following steps:
- __STEP 1__: Solve the pretext task i.e. `simclr.py`
- __STEP 2__: Perform the clustering step i.e. `scan.py`
- __STEP 3__: Perform the self-labeling step i.e. `selflabel.py`

For example, run the following commands sequentially to perform our method on CIFAR10:
```shell
python simclr.py --config_env configs/your_env.yml --config_exp configs/pretext/simclr_cifar10.yml
python scan.py --config_env configs/your_env.yml --config_exp configs/scan/scan_cifar10.yml
python selflabel.py --config_env configs/your_env.yml --config_exp configs/selflabel/selflabel_cifar10.yml
```
Then you can run SwASUE:
```shell
python main.py --config_env configs/your_env.yml --config_exp configs/selflabel/selflabel_cifar10.yml
```


## Citation

This Repository makes use of the repository: [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification) Please consider citing their work:

```bibtex
@inproceedings{vangansbeke2020scan,
  title={Scan: Learning to classify images without labels},
  author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}

```
For any enquiries, please contact the main authors.

## License

This toolkit is released under the MIT license. Please see the [LICENSE]() file for more information.



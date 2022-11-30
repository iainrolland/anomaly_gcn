## Dependencies

    * tensorflow
    * spektral
    * numpy
    * scipy
    * sklearn
    * tqdm
    * argparse
    * logging

## Demo training - London InSAR dataset
### Train conventional GCN

    python training.py --model_dir experiments/LondonTimeSeries/demo_GCN
   This will train a GCN model, with outputs saved in the `experiments/LondonTimeSeries/demo_GCN` directory. Files saved on output include:
   * `VanillaGCN.h5` - the tensorflow model/weights
   * `val_loss_history.npy` - the loss curve (validation set) during training
   * `val_acc_histroy.npy` - the accuracy curve (validation set) during training
   * `prob.npy` - the output probabilities (for all graph nodes regardless of train/validation/test)
   * `train.log` - log file, includes details such as misclassification AUROC/AUPR for associated uncertainty metrics
 ### Train S-BGCN

    python training.py --model_dir experiments/LondonTimeSeries/demo_S-BGCN
   
 This will train a S-BGCN model (equivalent to S-GCN model except that dropout applied at inference time to compute aleatoric/epistemic uncertainty) and save files into `experiments/LondonTimeSeries/demo_S-BGCN` directory with the addition that:
 * `alpha.npy` - is file containing Dirichlet concentration parameters for each graph node (i.e. of shape `(#nodes, #classes)`)
 ### Train S-BGCN-T

    python training.py --model_dir experiments/LondonTimeSeries/demo_S-BGCN-T
   
 This will train a S-BGCN-T model. If you look at the values in the `experiments/LondonTimeSeries/demo_S-BGCN-T/params.json`, you will notice that it points to `experiments/LondonTimeSeries/demo_GCN/prob.npy` as its teacher model. __You will get an error if you try to train this model before the GCN model__.
  ### Train S-BGCN-T-K

    python training.py --model_dir experiments/LondonTimeSeries/demo_S-BGCN-T-K
   
 This will train a S-BGCN-T-K model. If you look at the values in the `experiments/LondonTimeSeries/demo_S-BGCN-T/params.json`, you will notice that it points to `experiments/LondonTimeSeries/demo_GCN/prob.npy` as its teacher model. __You will get an error if you try to train this model before the GCN model__. The GKDE prior is computed on the fly using the stored adjacency matrix and training mask files.
 ### `models.py`
 Contains the classes describing each of the `tensorflow` models including:
 * GCN
 * S-BGCN
 * S-BGCN-T
 * S-BGCN-T-K
 * DPN-RS (https://github.com/JakobCode/dpn_rs)
	 * `@article{gawlikowski2022Andvanced,  
title={An Advanced Dirichlet Prior Network for Out-of-distribution Detection in Remote Sensing},  
author={Gawlikowski, Jakob and Saha, Sudipan and Kruspe, Anna and Zhu, Xiao Xiang},  
journal={IEEE Transactions in Geoscience and Remote Sensing}  
year={2022},  
publisher={IEEE}  
}`
### `losses.py`
Contains custom loss functions used for training
### `config_files` - directory
Contains example parameter files. To run your own experiments, you can take the contents of these files and place it in a directory within `experiments` and name the file `params.json`. Notes, each experiment should have its own `params.json` file contained within its own directory.
### `data_lib` - package
Contains the data loading functionality for each of the associated datasets. Note, some have `config.py` files which contain a `DATA_DIR` variable to point to a copy of the dataset on disk. Note, quite a few of the datasets are not themselves contained within this repository. Their code is still included for the purposes of providing details on specific implementation/preparation details. The `london_time_series` data is included for the purposes of code demonstration. For details regarding the other datasets, please contact the lead author directly.
### `spektral_datasets` - package
Interfaces between the data loaded by `data_lib` package and the `spektral` library. In `spektral_datasets/__init__.py` there is a `get_dataset()` function. This shows the valid `data` values for `params.json` settings i.e.:
* "AirQuality"
* "BeirutDataset"
* "HaitiDamage"
* "HoustonDatasetMini"
* "LondonTimeSeries"
* "SoilMoisture"
### `experiments` - directory
Specific folder for storing each experiment. Each experiment should be given its own folder with its own `params.json` file. See `experiments/LondonTimeSeries/demo_S-BGCN-T` for an example of how one might construct an experiment. Values in `experiments/LondonTimeSeries/demo_S-BGCN-T/params.json` can be adjusted to control experiment settings.
### `params.json` files
A parameters `.json` should always be named `params.json` and will have a subset of the following in it:
* `model`: defines which model is trained, should be one of:
	* "GCN",
	* "Drop-GCN"
	* "S-GCN"
	* "S-BGCN"
	* "S-BGCN-T"
	* "S-BGCN-K"
	* "S-BGCN-T-K"
	* "S-BMLP"
	* "AE"
	* "DPN-RS"
* `gpu_list`: if your machine has GPUs, you can specify which to use here with `[]` representing no GPUs, `[0]` denoting the first GPU should be used etc.
* `seed`: random seed for experiment (affects train/validation/test splits as well as model weight initialisations etc.)
* `k`: number of neighbours used when computing k-nearest neighbours graph
* `train_ratio`: percentage of samples to use for training e.g. `0.5` denotes 50%
* `val_ratio`: percentage of samples to use for validation (used for early-stopping) e.g. `0.2` denotes 20%
* `channels`: number of features/channels output in GCN hidden layers
* `alpha_prior_coefficient` (only used for `S-BGCN-T-K`) the $\lambda_K$ coefficient which weights the GKDE prior contribution to total loss
* `teacher_coefficient` (only used for `S-BGCN-T`/`S-BGCN-T-K` etc.) the $\lambda_T$ coefficient which weights the teacher network's contribution to the total loss function
* `teacher_file path` (only used for `S-BGCN-T`/`S-BGCN-T-K` etc.) points to a pre-trained GCN model's output probabilities e.g. `experiments/LondonTimeSeries/demo_GCN/prob.npy`
* `ood_classes`: controls whether to hold out specific classes to act as OOD for anomaly detection analysis (e.g. `[]` denotes do not hold any classes out, `[2, 3]` denotes that class `2` and `3` are hidden from training)
* `data`: controls which dataset experiment involves, one of:
	* `"AirQuality"`
	* `"BeirutDataset"`
	* `"HaitiDamage"`
	* `"HoustonDatasetMini"`
	* `"LondonTimeSeries"`
	* `"SoilMoisture"`
* `learning_rate`: the learning rate of the Adam optimiser used for model weight optimisation
* `l2_loss_coefficient`: model weight regularisation coefficient
* `epochs`: maximum number of epochs to train model for
* `patience`: number of consecutive epochs without validation loss improvement which triggers early-stopping

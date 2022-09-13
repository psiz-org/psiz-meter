![PsiZ logo](docs/img/full_logo_300.png)

# Psiz Meter

---
**WARNING:** This package is pre-release and the API is not stable. All APIs are subject to change and all releases are alpha. It is recommended you pin the repository version if you use in your own work. 

---

## Purpose
Provides python scripts and tools for evaluating target models relative to human behavior and PsiZ embeddings.

The repository currently contains scripts for evaluating embeddings for one dataset: the ILSVRC 2012 validation dataset. Additional datasets may be added in the future.

## Installation
1. Clone the GitHub repository to your local machine: `git clone https://github.com/psiz-org/psiz-meter`.
2. Install the cloned respository to your local virtual enviroment: `pip install /path/to/cloned/psiz-meter/`
3. (optional) Download large files hosted on [OSF psiz-meter](https://osf.io/dpt2f/).

## Datasets
Scripts, intermediate assets, and final assets for each dataset can be found in the `dataset` directory. For a given dataset, assets include PsiZ models (`psiz_models/`) and target model embeddings (`target_embeddings/`). 

### ILSVRC 2012 Validation

#### Roads & Love, 2021 CVPR

To reproduce the results presented in Roads & Love (2021), execute the following scripts in `datasets/ilsvrc2012_val/scripts/`:
1. (optional) Assemble embeddings for all target models by executing `assemble_target_embeddings.py`. The pre-assembled target embeddings are hosted on [OSF psiz-meter](https://osf.io/dpt2f/), so you can skip this step if you have already downloaded the necessary files. The scripts for assembling DeepCluster embeddings is more involved and can be found on the DeepCluster GitHub page.
2. Compute *triplet accuracy* for all target models by executing `cvpr2021_triplet_accuracy.py`.
3. Output LaTeX table of triplet accuracies by executing `cvpr2021_table2.py`.
3. Compute *embedding correlation* for all target models by executing `cvpr2021_embedding_correlation.py`.
4. Output LaTex table of embedding correlations by executing `cvpr2021_table3.py`.

Please see the following paper for more details on how the models were evaluated:
```
@InProceedings{Roads_Love_2021:CVPR,
    title     = {Enriching ImageNet with Human Similarity Judgments and Psychological Embeddings},
    author    = {Brett D. Roads and Bradley C. Love},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    month     = {6},
    pages     = {3547--3557}
    doi       = {10.1109/CVPR46437.2021.00355},
}
```

## Notes
* For all scripts, you will need to update the variabel `fp_project` to reflect the location of the repository directory on your machine.
* Running the scripts is computationally expensive. To make the pipeline more user-friendly, some intermediate results are saved to disk (i.e, target embeddings, triplet observations, triplet accuracy results, embedding correlation results).
* Running the scripts as-is will overwrite existing results in the database. If you would like to start a new database of results (and keep the original database unmodified), specify a new filename for the variable `fp_db` in the scripts (e.g., `db_cvpr2021_new.txt` instead of `db_cvpr2021.txt`).
* This repository does not include dataset files. For example, if you want to derive ILSVRC 2012 embeddings using a particular computer vision model, you must source the image files on your own.
* When you load the PsiZ models included in this repository, you will get a TensorFlow warning that you can safely ignore: "WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected when loading Keras model. Please ensure that you are saving the model with model.save() or tf.keras.models.save_model(), *NOT* tf.saved_model.save(). To confirm, there should be a file named "keras_metadata.pb" in the SavedModel directory." This warning is generated because the models were saved with an older version of TF, but they still work fine with the TF version specified by this repository.

## Resources
* Official Psiz Documentation: [psiz.readthedocs.io/en/latest](https://psiz.readthedocs.io/en/latest/)

## Contribution Guidelines
If you would like to contribute please see the [contributing guidelines](CONTRIBUTING.md).

This project uses a [Code of Conduct](CODE.md) adapted from the [Contributor Covenant](https://www.contributor-covenant.org/)
version 2.0, available at <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>.

## Licence
This project is licensed under the Apache Licence 2.0 - see LICENSE file for details.

# [WACV 2026] MergeSlide: Continual Model Merging and Task-to-Class Prompt-Aligned Inference for Lifelong Learning on Whole Slide Images

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-2401.xxxxx-b31b1b.svg" alt="arXiv">
  <img src="https://img.shields.io/badge/WACV-2026-blue.svg" alt="WACV2026">
</p>

> Doanh C. Bui (NAIST)*, Ba Hung Ngo (CNU), Hoai Luan Pham (NAIST), Khang Nguyen (UIT), Mai K. Nguyen (ETIS), Yasuhiko Nakashima (NAIST)

<img width="1428" height="580" alt="{285A97E6-9C0D-485C-B01E-DB1C802FDCEE}" src="https://github.com/user-attachments/assets/c16c4012-4789-457f-a3f3-f21482eb7bbd" />

## 📌 Status Updates

![update](https://img.shields.io/badge/2025--xx--xx-TODO-blue) Update checkpoints.

![update](https://img.shields.io/badge/2025--xx--xx-TODO-blue) Clean the code.

![update](https://img.shields.io/badge/2025--xx--xx-TODO-blue) Released pre-processed TCGA WSI features.

![update](https://img.shields.io/badge/2025--11--15-DONE-green) Release source code.

![update](https://img.shields.io/badge/2025--11--11-DONE-green) Accepted by **WACV2026**.

## 1. Requirement

- transformers=4.56.2
- torch=2.5.1
- torchaudio=2.5.1
- torchvision=0.20.1
- tqdm=4.67.1
- transformers=4.56.2

## 2. Datasets

### 2.1. Access Datasets

We use a stream of six datasets TCGA-BRCA, TCGA-NSCLC, TCGA-RCC, TCGA-ESCA, TCGA-TGCT and TCGA-CESC in this study.

For dataset preparation, you may need to download the WSIs from the TCGA portal and process them (patch extraction + feature extraction using TITAN’s vision encoder). If you are not familiar with this procedure, please refer to 20251020_WSI_processing.ipynb.

For convenience, we also provide pre-processed features that can be used directly with the scripts below *(will update later)*.

- Data annotation: **TBD** (will be a google drive's link here)
- WSI features: **TBD** (will be a google drive's link here)

### 2.2. Data Preparation

For dataset preparation, ESCA, TGCT, and CESC have a slightly different format compared with BRCA, NSCLC, and RCC. Therefore, two separate Python classes are defined for these groups. However, for both training and inference, we only need to prepare the data paths in `datasets.py` as follows (in `Sequential_Generic_MIL_Dataset` class):

```[python3]
datasets = [Generic_MIL_Dataset(csv_path='/path/to/dataset/wsi_dataset_annotation/tcga_brca/tcga_brca_subset.csv.zip', data_dir='/path/to/dataset/TCGA-BRCA_processed/features/', shuffle=False, seed=0, print_info=True, label_dict={'IDC': 0, 'ILC': 1}, patient_strat=False, ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT']), 
                Generic_MIL_Dataset(csv_path='/path/to/dataset/wsi_dataset_annotation/tcga_rcc/tcga_kidney_subset.csv.zip', data_dir='/path/to/dataset/TCGA-RCC_processed/features/', shuffle=False, seed=0, print_info=True, label_dict={'CCRCC': 0, 'PRCC': 1, 'CHRCC': 2}, patient_strat=False, ignore=[]), 
                Generic_MIL_Dataset(csv_path='/path/to/dataset/wsi_dataset_annotation/tcga_nsclc/tcga_lung_subset.csv.zip', data_dir='/path/to/dataset/TCGA-NSCLC_processed/features/', shuffle=False, seed=0, print_info=True, label_dict={'LUAD': 0, 'LUSC': 1}, patient_strat=False, ignore=[]), 
                Generic_MIL_Dataset2(data_dir='/path/to/dataset/TCGA-ESCA_processed/features/', label_dict={0: 0, 1: 1}), 
                Generic_MIL_Dataset2(data_dir='/path/to/dataset/TCGA-TGCT_processed/features/', label_dict={0: 0, 1: 1}), 
                Generic_MIL_Dataset2(data_dir='/path/to/dataset/TCGA-CESC_processed/features/', label_dict={0: 0, 1: 1})]
    
split_dirs = ['/path/to/dataset/wsi_dataset_annotation/tcga_brca',
                '/path/to/dataset/wsi_dataset_annotation/tcga_rcc', 
                '/path/to/dataset/wsi_dataset_annotation/tcga_nsclc', 
                '/path/to/dataset/wsi_dataset_annotation/tcga_esca', 
                '/path/to/dataset/wsi_dataset_annotation/tcga_tgct', 
                '/path/to/dataset/wsi_dataset_annotation/tcga_cesc']
```

In this setup, `/path/to/dataset` refers to the root directory of the dataset. The wsi_dataset_annotation file and all feature directories under `/TCGA-*_processed/features` are provided in **Section 2.1: Access Datasets**.

## 3. Implementation

As described in the paper, we first define class-aware prompts to describe a set of class labels (**Class-aware Prompt Design**), get their embeddings using TITAN's text encoder, and train each model on its corresponding TCGA task using the pre-trained weights of TITAN’s slide aggregator (**Per-task finetuning**). We then merge these models using a continual model-merging method (**Model merging**).

**Note:** You may need to be granted to access TITAN pre-trained slide aggregator by yourself. Please visit https://huggingface.co/MahmoodLab/TITAN.

### 3.1. Class-aware Prompt Design

For the six tasks in this study, please refer to `prompts_zeroshot.py`. You may design class-aware prompts for your new task by following the templates provided in that file.

### 3.2. Per-task Finetuning

Run the below python script to perform per-task finetuning:

```
python train_random_sampling.py --save_dir /path/to/finetuned/checkpoints
```

where `/path/to/finetuned/checkpoints` is the directory where you want the model checkpoints to be stored.

### 3.3. Model Merging

Run the following Python script to perform model merging:

```
python opcm_mergeslide.py --num_tasks 6 
--src_finedtuned_checkpoints /path/to/finetuned/checkpoints 
--des_merged_checkpoints /path/to/merged/checkpoints/
```

Merged checkpoints are stored in `/path/to/merged/checkpoints/` (All tasks only use one checkpoint for later inference).

Note: Because we perform 10-fold cross-validation, the script generates 10 folders named `/path/to/merged/checkpoints/_fold_*`. Each folder contains num_tasks merged checkpoints (6 tasks in this study), representing the accumulated model state after each task. These intermediate checkpoints are used only for evaluating continual learning metrics such as forgetting, BWT, and FWT.

```
[/path/to/merged/checkpoints/]_fold_0
|___merged_weight_opcm_random_sampling_fold_0_task_0.pth
|___merged_weight_opcm_random_sampling_fold_0_task_1.pth
|___merged_weight_opcm_random_sampling_fold_0_task_2.pth
|___...
|___merged_weight_opcm_random_sampling_fold_0_task_6.pth
```

If you do not need to compute forgetting, BWT, or FWT, only the final checkpoint`merged_weight_opcm_random_sampling_fold_0_task_6.pth`
is required for inference on all 6 tasks in this study.

### 3.3. Evaluation

The evaluation is designed for CLASS-IL and TASK-IL scenario.

For CLASS-IL, if we only need Accuracy, Balanced Accuracy, Macro/Weighted F1, Precicion, Recall for all tasks after training the last task, just run:

```
python test_classIL_task_prompt.py --save_dir /path/to/finetuned_checkpoints 
--merge_model_path /path/to/merged/checkpoints/
```

If we need Forgetting, BWT, FWT:

```
python test_classIL_task_prompt_other_metrics.py --save_dir /path/to/finetuned_checkpoints 
--merge_model_path /path/to/merged/checkpoints/
```

For TASK-IL scenario, just run:

```
python test_taskIL.py --save_dir /path/to/finetuned_checkpoints 
--merge_model_path /path/to/merged/checkpoints/
```

**Note 1:** we load `/path/to/finetuned_checkpoints` only to extract the class-aware prompts from the per-task finetuned checkpoints (these checkpoints remain completely frozen during training; that is, the class-aware prompt embeddings do not change from TITAN to the per-task finetuning stage). We do this purely for convenience. A more efficient implementation could be added later, where only the class-aware prompts are provided directly, eliminating the need to load the per-task finetuned checkpoints and avoiding potential confusion.

**Note 2:** For the main results, we report balanced accuracy. However, for metrics such as FGT, BWT, and Forgetting, we use standard accuracy for their calculation.

## 4. Acknowledgement

To complete this study, we were inspired by the following code bases:

- [TITAN](https://github.com/mahmoodlab/TITAN)
- [FusionBench](https://github.com/tanganke/fusion_bench)
- [CATE](https://github.com/HKU-MedAI/CATE)

Once again, we sincerely thank the authors of these projects for their tremendous effort and contributions, which allowed us to stand on the shoulders of giants.

## 5. Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{
    bui2026merge,
    title={MergeSlide: Continual Model Merging and Task-to-Class Prompt-Aligned Inference for Lifelong Learning on Whole Slide Images},
    author={Doanh C. Bui, Ba Hung Ngo, Hoai Luan Pham, Khang Nguyen, Maï K. Nguyen, Yasuhiko Nakashima},
    booktitle={The IEEE/CVF Winter Conference on Applications of Computer Vision},
    year={2026},
}
```
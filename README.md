# ConLoRA

ConLoRA is a platform for distributed fine-tuning of large models in a fully decentralized federated learning scenario. This platform allows you to perform fine-tuning of large models within a completely decentralized network. In particular, ConLoRA addresses inherent error issues in decentralized federated learning by providing an option to mitigate the consensus error amplification effect caused by LoRA. You can freeze the A matrix during training to further reduce this error.

## How to Use

### 1. Download

Clone the repository to your local folder:

```bash
git clone https://github.com/LuYuPromax/ConLoRA.git
```

### 2. Create environment

Create a new Conda environment with Python 3.10.10:

```
conda create -n env_name python==3.10.10
```

Then install the related libraries

```
pip install -r requirement.txt
```

### Datasets

Currently, ConLoRA supports some GLUE datasets and the GSM8k dataset. To use them, follow these steps:

1. Go to the `utils` folder.

2. Modify the paths in `glue_split.sh` and `gsm8k_split.sh` to match your local dataset storage.

3. Run the following commands to split the datasets:

##### GLUE

```
chmod +x glue_split.sh
./glue_split.sh
```

##### GSM8K

```
chmod +x gsm8k_split.sh
./gsm8k_split.sh
```

To view the data distribution of different clients, run the view_split.sh script.

### Finetune

To start the federated training process, run:

```
chmod +x train.sh
./train.sh
```

Parameter Description

- **`--model_checkpoint` (str)**: Path to the pre-trained model checkpoint.  
- **`--dataset_path_template` (str)**: Template for client dataset paths (`{i}` is replaced by client number).  
- **`--val_dataset_path_template` (str)**: Template for client validation dataset paths.  
- **`--num_clients` (int)**: Number of clients participating in federated learning.  
- **`--lora_r` (int)**: LoRA rank for LoRA layers.  
- **`--lora_alpha` (int)**: LoRA alpha for LoRA layers.  
- **`--target_modules` (str)**: Comma-separated list of target modules for LoRA layers.  
- **`--training_type` (str)**: Type of training (`LoRA` or `ConLoRA`).  
- **`--dataset_type` (str)**: Dataset type (`sst2`, `mnli`, or `qnli`).  
- **`--name` (str)**: Name used to generate the weight matrix.  
- **`--num_rounds` (int, default: 256)**: Number of federated learning rounds.  
- **`--batch_size` (int, default: 128)**: Batch size for each client.  
- **`--log_path` (str, default: "federated_training.log")**: Path to save the log file.  

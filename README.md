# Anatomically-Informed Cerebral Artery Labelling

Perform node labelling on a cerebrovasculature graph.
\
Datasets must be in BIDS or SPARC format using MRAtoBG substructure.

🔍 **Note:** Ensure the graph has no connectivity errors and is roughly in standard orientation. For further details please see:

`Shen, J., Maso Talou, G. (2025)`

🔍 **Note:** This repo contains code and models derived from the [GNN-ART-LABEL](https://github.com/clatfd/GNN-ART-LABEL) repo. For further details please see:

`Chen, T., You, W., Zhang, L., Ye, W., Feng, J., Lu, J., ... & Guan, S. (2024). Automated anatomical labeling of the intracranial arteries via deep learning in computed tomography angiography. Frontiers in Physiology, 14, 1310357.`

---

## ⚠️ System Requirements

Please ensure your system meets these requirements for full compatibility:

- **OS**: Ubuntu 20.04
- **Python**: 3.7
- **GPU**: NVIDIA V100
- **CUDA**: Version 11.7

---

## 🚀 Quick Start

### 1. Clone the Repository

Create a folder to store this MRAtoBG repo, for example:

```bash
/user/repos
```

Navigate to the above folder and clone this repository:

```bash
cd /user/repos
git clone https://github.com/ABI-Animus-Laboratory/MRAtoBG_node_labelling
```

Create an environment variable to this repo, for example:

```bash
export MRAtoBG_NODE_LABELLING_PATH=/user/repos/MRAtoBG_node_labelling
```

### 2. Setup Python Environment

Create and activate a new virtual environment:

```bash
python -m venv $MRAtoBG_NODE_LABELLING_PATH/venv
source $MRAtoBG_NODE_LABELLING_PATH/venv/bin/activate
pip install -r $MRAtoBG_NODE_LABELLING_PATH/requirements.txt
```

💡If you encounter issues when installing packages, ensure you have updated pip.


### 3. Configure Your Study

Create an environment variable to the root of your study dataset folder, for example:

```bash
export DATASET_ROOT_PATH=/user/studies/study0
```

Download the `MRAtoBG_config.json` file from [here](https://github.com/ABI-Animus-Laboratory/MRAtoBG_brain_vessel_segmentation/blob/main/MRAtoBG_config.json) and store it at `$DATASET_ROOT_PATH/code`.

Edit the `MRAtoBG_config.json` configuration file as below:

```json
{
  "dataset_structure": "your dataset structure",
  "dataset_root_path": "your dataset root path",
  "do_ICA_init": "1",
  "do_FPNL": "1",
  "node_labelling_dist_thresholds_root": "see below",
  "node_labelling_ICA_init_model": "see below",
  "node_labelling_ICA_init_model_path": "see below",
}
```

* Replace `"your dataset structure"` with "BIDS" or "SPARC", depending on your dataset structure.
* Replace `"your dataset root path"` with your dataset root path.
* Set `"do_ICA_init"` to "1" to perform **ICA key node initialisation**, or "0" otherwise.
* Set `"do_FPNL"` to "1" to perform **node labelling using the FPNL algorithms for remaining CoW key node**, or "0" otherwise.

🔍 **Note:** Use the following settings to use existing reference artery segment length thresholds (dist_thresholds) and a pre-trained [GNN-ART-Label](https://github.com/clatfd/GNN-ART-LABEL) model already available in this repo:

* "node_labelling_dist_thresholds_root" = `$MRAtoBG_NODE_LABELLING_PATH/config`
* "node_labelling_ICA_init_model" = "GNNART_7b_bestValLoss"
* "node_labelling_ICA_init_model_path": `$MRAtoBG_NODE_LABELLING_PATH/models/icafe 7b/model66000-0.3533-0.9653-0.8410-0.0776.ckpt`

### 4. Run the Module

Select an available GPU with index `XXX`, and run the command below:

```bash
CUDA_VISIBLE_DEVICES=XXX \
$MRAtoBG_NODE_LABELLING_PATH/venv/bin/python \
$MRAtoBG_NODE_LABELLING_PATH/src/run_MRAtoBG_node_labelling.py \
$DATASET_ROOT_PATH/code/MRAtoBG_config.json
```

💡If the `_node_labelling_ICA_init_raw.html` file (under `MRAtoBG/node_labelling` for each subject) does not show any non-black nodes (and you get an error saying an ICA key node does not exist), it is likely that you did not execute the code on a GPU. Resolve this issue.
\
💡If you wish to manually label the ICA key nodes to bypass `ICA key node missing` errors, manually label the ICA key nodes in the `_node_labelling_ICA_init.csv` file (under `MRAtoBG/node_labelling` for each subject) by referring to the instructions in the next section, and then run this module with `"do_ICA_init"` set to "0".

---

## 📝 Notes

- Ensure your dataset follows the BIDS or SPARC dataset structures.
- Outputs will be written to the `MRAtoBG/node_labelling` subfolder.

# Encoding instructions

For utilization of the code, there are only two files that need to be run:

1. **Training models and storing results using `train.py`**
2. **Retrieving model and generating plots using `generate_plots.py`**

## 1. Training the Model (`train.py`)

### **Usage**
Run the training script with:

```bash
python src/train.py --mode <mode> --custom-split --file-identifier "<identifying text>" --max-step <max_step> --step-size <step_size>
```

### **Arguments**
| Argument             | Description |
|---------------------|-------------|
| `--mode`           | Prediction type: `speed`, `position`, `distance`, `rel_angle`, `abs_angle` |
| `--custom-split`   | Use custom train-test split based on episode data (optional) |
| `--file-identifier` | Custom identifier for the output pickle file (mandatory) |
| `--max-step`       | Maximum time step to predict (default: 500) |
| `--step-size`      | Step size for time step range (default: 10) |

`position` is relative position, while `distance` is the true distance from origin.

### **Example**
```bash
python src/train.py --mode distance --custom-split --file-identifier "test_run" --max-step 500 --step-size 10
```
This generates a pickle file in `out/` containing model predictions and evaluation metrics.

### **File Dependencies**
- **`models.py`**: Contains the model definition and evaluation functions.
- **`preprocessing.py`**: Handles data preprocessing and feature extraction.
- **Input Files**: You must specify `file_names_hstates` and `file_names_scalars` within `train.py`. These are the files from the recording windows that contain the neural activity and scalar data, respectively.

---

## 2. Generating Plots (`generate_plots.py`)

Once training is complete, use `generate_plots.py` to visualize results.

### **Usage**
Run the cells in the notebook.


### **File Dependencies**
- **`plots.py`**: Contains helper functions for plotting.
- **Pickle Files**: Generated from `train.py`, containing model results. You must decide which files to plot analysis on. 

### **Available Plots**
- **`plot_rmse(filename)`**: Compares RMSE between naive and trained models.
- **`plot_contributions(filename)`**: Visualizes neuron contributions.
- **`plot_top_contributors(filenames, k_percent)`**: Analyzes top neuron contributions across multiple files.
- **`plot_rolling_contributions(filename, window_size, k_percents)`**: Shows rolling contributions over time.

---

## **Project Structure**
```
|
├── train.py            # Main training script
├── generate_plots.py   # Plot generation script
├── models.py           # Model definition and training logic
├── preprocessing.py    # Data preprocessing functions
├── plots.py            # Helper functions for visualization
│
├── out/                    # Output directory for pickle files
├── data/                   # Raw and processed data files
├── README.md               # Project documentation
```


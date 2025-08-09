# Bibliometrics: ML training, hyperparameter tuning, and walk forward cross validation

---

## 📊 Results
<!-- 
Insert key performance visualizations here (e.g., accuracy curves, PR curves, confusion matrices).
Example:
![Validation Accuracy Over Time](./results/val_accuracy.png)
![Precision-Recall Curve](./results/precision_recall.png)
-->
*(Results will be added following model evaluation.)*

---

## Project Context
This module is part of the **Academic Abstracts Project**, a large-scale pipeline for acquiring, integrating, and analyzing scholarly publication data.  
The **training component** focuses on building predictive models that leverage **semantic embeddings** and **bibliometric features** to classify research outputs based on citation performance.

---

**Key Design Principles:**
- **Time-aware evaluation** to prevent future data leakage.
- **Reference context modeling** to capture citation network influence.
- **Scalable data handling** for large embedding datasets.
- **Automated hyperparameter search** for optimal architecture selection.

---

## Objectives
- **Load and preprocess** high-dimensional embedding datasets for supervised learning.
- **Construct temporal, reference-aware feature sets** to capture citation network context.
- **Perform hyperparameter optimization** using Bayesian search.
- **Evaluate models** under a **time-aware cross-validation** framework to ensure realistic generalization performance.
- **Log and store** all results for reproducibility and portfolio presentation.

---

## Current Capabilities

### 1. Data Loading & Preprocessing
- Reads **Parquet** feature datasets generated in earlier pipeline stages.
- Uses **PyArrow** and **DuckDB** for efficient, columnar data access.
- Implements **reference-based feature construction**:
  - For each focal paper, retrieves embeddings of up to `N_REF` referenced works.
  - Concatenates reference embeddings with the focal paper’s own embedding.
- Produces `(X, y)` pairs:
  - **X**: Flattened embedding vectors.
  - **y**: Binary label (`higher_than_median_year`).

---

### 2. TensorFlow Dataset Pipeline
- Implemented in `src/load_data.py`.
- Uses `tf.data.Dataset.from_generator` for **streaming large datasets** without exhausting memory.
- Supports **on-disk caching** for faster re-runs.
- Automatic **prefetching** for GPU/TPU efficiency.
- Configurable parameters:
  - `n_back` (number of references to include)
  - `n_features` (embedding dimensions)
  - `batch_size` (auto-calculated based on available RAM)

---

### 3. Model Architecture & Hyperparameter Tuning
- Implemented in `src/tune.py`.
- **Keras Tuner** with **Bayesian Optimization** for efficient search.
- Search space includes:
  - L2 regularization strength
  - Learning rate
  - Exponential decay rate
  - Number of convolutional layers
  - Filters per layer
  - Kernel sizes
  - Optimizer type
- Model design:
  - Input reshaped to `(N_REF+1, embedding_dim)`
  - 1D Convolutional layers + Batch Normalization + ReLU
  - Flatten + Dense softmax output
- Loss: `sparse_categorical_crossentropy`
- Primary metric: `accuracy` (additional metrics computed post-training)

---

### 4. Temporal Cross-Validation
- Splits data into **train / validation / test** sets based on publication date.
- Uses a **sliding window** (`CV_delta` years) to simulate real-world forecasting.
- Prevents **future data leakage** into training.
- For each time slice:
  1. Load train & validation sets.
  2. Run hyperparameter search (first slice only).
  3. Retrain best model on full training set.
  4. Evaluate on test set.
  5. Save metrics to `./test_results_dir/`.

---

### 5. Early Stopping & Trial Pruning
- **EarlyStopping** on validation accuracy to prevent overfitting.
- **Custom callback** (`StopIfUnpromisingTrial`) to terminate trials early if they are unlikely to outperform the current best.
- Reduces computational overhead during tuning.

---

### 6. Metrics & Outputs
- Saves:
  - Best model (`.h5` format)
  - CSV metrics per test slice:
    - Loss
    - Accuracy
    - Balanced accuracy
    - Precision
    - Recall
- Logs training progress to `logs/` for reproducibility.

---

## Example Workflow

```bash
# 1. Ensure integrated & feature datasets are prepared
#    (Run earlier pipeline stages first)

# 2. Configure training in config/config.toml
# Example:
# [data]
# database_loc = "~/data/abstracts_features/ACS"
# Nembeddings = 384
#
# [train]
# start_year = 2010
# end_year = 2023
# start_train_size = 5
# val_size = 1
# test_size = 1
# CV_delta = 1
#
# [model]
# model_name = "conv_ref_model"
#
# [log]
# file = "INFO"
# console = "INFO"

# 3. Run the training & tuning process
python src/tune.py
```

---

## Reproducibility & Best Practices
- **Deterministic seeds** set for TensorFlow and NumPy.
- **Config-driven** architecture for easy experiment replication.
- **Separation of concerns**:
  - Data loading (`load_data.py`)
  - Model tuning (`tune.py`)
  - Config management (`utils/load_config.py`)
- **Logging** at both console and file level for traceability.

---

## Next Steps (Planned Enhancements)
- Incorporate **additional bibliometric features** (e.g., citation velocity, h-index).
- Experiment with **transformer-based architectures** for sequence modeling of references.
- Extend evaluation to **multi-class citation impact prediction**.
- Integrate **model explainability** (e.g., SHAP, LIME) for feature importance analysis.

---

## Author
**Felix Noble**  
Contact: `felix.noble@live.co.uk`  
LinkedIn: *https://www.linkedin.com/in/felix-noble-6901b117b/*  

---

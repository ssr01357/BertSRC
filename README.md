# BertSRC
A repository implementation of the "BertSRC: BERT-based Semantic Relationship Classification" paper.

# Requirements
* Python >= 3.8
* Packages: 
```
pip install -r requirements.txt
```

# Train
Train BertSRC train dataset
```
python train.py
```

# Test
Test BertSRC test dataset. Train must be run before the test.
```
python test.py
```

# Notebooks
* kfold.ipynb
  * The notebook is an example of kfold evaluation for experiments mentioned in the paper.

* chemprot
  * The directory contains a chemprot dataset and a notebook that shows the results of the chemprot experiment.
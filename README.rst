Environment Setting
==================

``pip install -e .``
``pip install cox``

Train classifier (e.g. Pneumonia classifier)
==================
``python train_robust_classifier.py --mimic_path /PATH/TO/MIMIC/DIRECTORY --target labels "Pneumonia" "No Finding" --balance_labels True``
Right now, only binary classification (ie, two targets) is possible.


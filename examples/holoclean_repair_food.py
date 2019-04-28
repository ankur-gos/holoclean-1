import sys
sys.path.append('../')
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *
from repair.featurize.ratio_constraint_featurizer import RatioConstraintFeaturizer


# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0.7,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.3,
    nb_cor_strength=0.3,
    epochs=5,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=10,
    batch_size=32,
    verbose=True,
    timeout=3*60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=True
).session

# 2. Load training data and denial constraints.
hc.load_data('food', '../testdata/food.csv')
hc.load_dcs('../testdata/food_constraints.txt')
hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
#featurizers = [
#    InitAttrFeaturizer(),
#    OccurAttrFeaturizer(),
#    FreqFeaturizer(),
#    ConstraintFeaturizer(),
#]

featurizers = [
    FreqFeaturizer(),
    RatioConstraintFeaturizer()
]


hc.repair_errors(featurizers)

# 5. Evaluate the correctness of the results.
hc.evaluate(fpath='../testdata/food_clean.csv',
            tid_col='tid',
            attr_col='attribute',
            val_col='correct_val')

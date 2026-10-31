"""
portable_model.py  -  save & load trained models WITHOUT version problems.

Why this file exists
--------------------
Pickling a scikit-learn / XGBoost object only works on the SAME library version
(otherwise: "XGBoostError: input stream corrupted"). So instead we convert every
model into plain numbers (tree structure, weights, support vectors) and write our
own tiny predict function. Result: the Streamlit app needs only numpy.

Used by:  the notebook (to_portable, portable_predict)   and   app.py (portable_predict)
"""
import json
import numpy as np


# ------------------------------------------------------------------ 1. convert a trained model
def _sklearn_tree(est):
    t = est.tree_
    return {"left": t.children_left.astype(np.int32), "right": t.children_right.astype(np.int32),
            "feature": t.feature.astype(np.int32), "threshold": t.threshold.astype(np.float64),
            "value": t.value[:, 0, 0].astype(np.float64), "strict": False}   # rule: x <= threshold -> go left


def _xgboost_trees(model):
    # Read the trees from XGBoost's own JSON dump (exact numbers, works on all XGBoost versions)
    dump = json.loads(model.get_booster().save_raw("json"))
    trees = []
    for t in dump["learner"]["gradient_booster"]["model"]["trees"]:
        left = np.array(t["left_children"], dtype=np.int32)
        trees.append({"left": left, "right": np.array(t["right_children"], dtype=np.int32),
                      "feature": np.array(t["split_indices"], dtype=np.int32),
                      "threshold": np.array(t["split_conditions"], dtype=np.float32),   # for leaves this holds the leaf value
                      "value": np.array(t["split_conditions"], dtype=np.float64),
                      "strict": True})                                                  # rule: x < threshold -> go left
    base = dump["learner"]["learner_model_param"]["base_score"]
    return trees, float(str(base).strip("[]"))


def to_portable(model):
    """Turn a trained model into plain numpy arrays / numbers."""
    kind = type(model).__name__
    if kind in ("LinearRegression",):
        return {"type": "linear", "w": model.coef_.astype(np.float64), "b": float(model.intercept_)}
    if kind == "GradientDescentRegressor":                           # our own class from the notebook
        return {"type": "linear", "w": np.asarray(model.w, dtype=np.float64), "b": float(model.b)}
    if kind == "DecisionTreeRegressor":
        return {"type": "tree", **_sklearn_tree(model)}
    if kind == "RandomForestRegressor":
        return {"type": "forest", "trees": [_sklearn_tree(e) for e in model.estimators_]}
    if kind == "SVR":
        return {"type": "svr", "sv": model.support_vectors_.astype(np.float64),
                "dual_coef": model.dual_coef_[0].astype(np.float64),
                "b": float(model.intercept_[0]), "gamma": float(model._gamma)}
    if kind == "XGBRegressor":
        trees, base = _xgboost_trees(model)
        return {"type": "xgb", "trees": trees, "base": base}
    raise ValueError(f"Unknown model type: {kind}")


# ------------------------------------------------------------------ 2. predict with the portable form
def _predict_tree(t, X32):
    node = np.zeros(len(X32), dtype=np.int64)          # every row starts at the root
    rows = np.arange(len(X32))
    while True:
        is_leaf = t["left"][node] == -1
        if is_leaf.all():
            return t["value"][node]
        x = X32[rows, np.where(is_leaf, 0, t["feature"][node])]
        go_left = (x < t["threshold"][node]) if t["strict"] else (x <= t["threshold"][node])
        node = np.where(is_leaf, node, np.where(go_left, t["left"][node], t["right"][node]))


def portable_predict(m, X):
    """Predict log(1 + yield) for an (n_rows, n_features) matrix."""
    X = np.asarray(X, dtype=np.float64)
    kind = m["type"]
    if kind == "linear":                                # Linear Regression / Gradient Descent
        return X @ m["w"] + m["b"]
    X32 = X.astype(np.float32)                          # tree libraries compare in float32
    if kind == "tree":                                  # Decision Tree
        return _predict_tree(m, X32)
    if kind == "forest":                                # Random Forest = average of trees
        return np.mean([_predict_tree(t, X32) for t in m["trees"]], axis=0)
    if kind == "xgb":                                   # XGBoost = base + sum of trees
        return m["base"] + np.sum([_predict_tree(t, X32) for t in m["trees"]], axis=0)
    if kind == "svr":                                   # RBF-kernel SVR
        sq_dist = (X ** 2).sum(1)[:, None] + (m["sv"] ** 2).sum(1)[None, :] - 2 * X @ m["sv"].T
        return np.exp(-m["gamma"] * sq_dist) @ m["dual_coef"] + m["b"]
    raise ValueError(f"Unknown model type: {kind}")

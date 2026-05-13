"""The five sample-identifiability models from the project plan (section 2.1).

All models implement a common pair-scoring interface::

    model.fit(X_train, patient_ids_train)          # X_train: (n_samples, n_features)
    scores = model.score_pairs(X, ia, ib)          # higher score == more likely same patient

``X`` is the model's representation matrix for whatever set of samples is being scored
(rows aligned to ``ia``/``ib`` indices).  The representation that should be fed in is
declared by the ``representation`` attribute and is one of:

  * ``"saap"``          -- binary SAAP detection matrix (Models 1 & 2)
  * ``"intensity_pca"`` -- PCA-reduced log-intensity matrix (Models 3 & 5)
  * ``"intensity_log"`` -- filtered/imputed log-intensity matrix (Model 4)

The orchestrator (``run.py``) is responsible for building these matrices with all
fitting (feature selection, scaling, PCA) done on training samples only.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - torch is optional
    _HAVE_TORCH = False


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _cosine_pairs(X, ia, ib):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return np.sum(Xn[ia] * Xn[ib], axis=1)


def _neg_euclidean_pairs(X, ia, ib):
    return -np.linalg.norm(X[ia] - X[ib], axis=1)


# --------------------------------------------------------------------------------------
# Model 1 -- SAAP Jaccard similarity (unsupervised baseline)
# --------------------------------------------------------------------------------------

class JaccardSAAP:
    representation = "saap"
    needs_training = False

    def __init__(self):
        self.name = "M1_SAAP_Jaccard"

    def fit(self, X_train, patient_ids_train):
        return self

    def score_pairs(self, X, ia, ib):
        A = X[ia] > 0
        B = X[ib] > 0
        inter = np.sum(A & B, axis=1).astype(float)
        union = np.sum(A | B, axis=1).astype(float)
        out = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        return out


# --------------------------------------------------------------------------------------
# Model 2 -- Population-frequency-weighted SAAP similarity
# --------------------------------------------------------------------------------------

class WeightedSAAP:
    """Weighted-Jaccard similarity over SAAP detection vectors.

    Per-feature weight ``w_i = -log10(freq_i + eps)`` where ``freq_i`` is the dbSNP
    population allele frequency of the SAAP (rare variants are more identifying).  If
    no external frequency is available for a feature, its corpus detection rate over the
    training samples is used as a proxy for ``freq_i``.  Optionally each feature weight is
    further scaled by a confidence factor derived from the peptidoform ``PValue``.
    """

    representation = "saap"
    needs_training = True

    def __init__(self, feature_freq=None, feature_pvalue=None, eps=1e-4, use_cosine=False):
        self.name = "M2_SAAP_weighted"
        self.feature_freq = feature_freq          # array aligned to columns of X, or None
        self.feature_pvalue = feature_pvalue      # array aligned to columns of X, or None
        self.eps = eps
        self.use_cosine = use_cosine
        self.w_ = None

    def fit(self, X_train, patient_ids_train):
        n_feat = X_train.shape[1]
        if self.feature_freq is not None and len(self.feature_freq) == n_feat:
            freq = np.asarray(self.feature_freq, dtype=float)
            # fall back to corpus rate where the dbSNP frequency is missing
            corpus_rate = X_train.mean(axis=0)
            freq = np.where(np.isnan(freq), corpus_rate, freq)
        else:
            freq = X_train.mean(axis=0)  # corpus detection rate as a rarity proxy
        w = -np.log10(np.clip(freq, 0, None) + self.eps)
        w = np.clip(w, 1e-6, None)
        if self.feature_pvalue is not None and len(self.feature_pvalue) == n_feat:
            pv = np.asarray(self.feature_pvalue, dtype=float)
            conf = np.clip(np.nan_to_num(pv, nan=0.0), 0.0, 20.0) / 20.0  # 0..1
            w = w * (0.25 + 0.75 * conf)  # never zero out a feature entirely
        self.w_ = w
        return self

    def score_pairs(self, X, ia, ib):
        w = self.w_ if self.w_ is not None else np.ones(X.shape[1])
        A = (X[ia] > 0).astype(float)
        B = (X[ib] > 0).astype(float)
        if self.use_cosine:
            Aw = A * np.sqrt(w)
            Bw = B * np.sqrt(w)
            num = np.sum(Aw * Bw, axis=1)
            den = np.linalg.norm(Aw, axis=1) * np.linalg.norm(Bw, axis=1)
            return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        inter = np.sum(w * (A * B), axis=1)
        union = np.sum(w * np.maximum(A, B), axis=1)
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


# --------------------------------------------------------------------------------------
# Model 3 -- Cosine / Euclidean similarity on PCA-reduced intensity features
# --------------------------------------------------------------------------------------

class SimilarityPCA:
    representation = "intensity_pca"
    needs_training = False

    def __init__(self, metric="cosine"):
        assert metric in ("cosine", "euclidean")
        self.metric = metric
        self.name = f"M3_PCA_{metric}"

    def fit(self, X_train, patient_ids_train):
        return self

    def score_pairs(self, X, ia, ib):
        if self.metric == "cosine":
            return _cosine_pairs(X, ia, ib)
        return _neg_euclidean_pairs(X, ia, ib)


# --------------------------------------------------------------------------------------
# Model 4 -- Random forest on pairwise feature differences
# --------------------------------------------------------------------------------------

class RandomForestPairs:
    """Random-forest classifier on the symmetric pairwise difference vector
    over a missingness-aware Representation-B feature space.

    The input matrix concatenates:
      1. filtered / imputed log-intensity features
      2. binary missingness indicators for those same retained features

    so the pairwise representation is still ``d = |A - B|``, but now it also contains
    0/1 differences that tell the forest whether a peptidoform was observed in one sample
    and missing in the other.
    """

    representation = "intensity_log_missing"
    needs_training = True

    def __init__(self, n_estimators=300, max_depth=None, min_samples_leaf=2,
                 neg_per_pos=5, seed=0):
        from sklearn.ensemble import RandomForestClassifier
        self.name = "M4_RandomForest"
        self.neg_per_pos = neg_per_pos
        self.seed = seed
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            n_jobs=-1, random_state=seed, class_weight="balanced", oob_score=True, bootstrap=True,
        )
        self.oob_score_ = None
        self.feature_importances_ = None

    @staticmethod
    def _diffs(X, ia, ib):
        return np.abs(X[ia] - X[ib])

    def _training_pairs(self, n, patient_ids):
        rng = np.random.RandomState(self.seed)
        pid = np.asarray(patient_ids)
        ia, ib, lab = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                ia.append(i); ib.append(j); lab.append(1 if pid[i] == pid[j] else 0)
        ia, ib, lab = np.array(ia), np.array(ib), np.array(lab)
        pos = np.where(lab == 1)[0]
        neg = np.where(lab == 0)[0]
        n_neg = min(len(neg), int(round(self.neg_per_pos * max(len(pos), 1))))
        if n_neg < len(neg):
            neg = rng.choice(neg, size=n_neg, replace=False)
        sel = np.concatenate([pos, neg])
        rng.shuffle(sel)
        return ia[sel], ib[sel], lab[sel]

    def fit(self, X_train, patient_ids_train):
        ia, ib, lab = self._training_pairs(X_train.shape[0], patient_ids_train)
        D = self._diffs(X_train, ia, ib)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf.fit(D, lab)
        self.oob_score_ = getattr(self.clf, "oob_score_", None)
        self.feature_importances_ = self.clf.feature_importances_
        return self

    def score_pairs(self, X, ia, ib):
        D = self._diffs(X, ia, ib)
        return self.clf.predict_proba(D)[:, 1]


# --------------------------------------------------------------------------------------
# Model 5 -- Contrastive metric learning (shallow embedding network)
# --------------------------------------------------------------------------------------

if _HAVE_TORCH:
    class _EmbedNet(nn.Module):
        def __init__(self, in_dim, emb_dim=64, hidden=128, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, emb_dim),
            )

        def forward(self, x):
            return self.net(x)


class ContrastiveEmbedding:
    """Shallow MLP trained with a contrastive loss so same-patient samples embed close
    together and different-patient samples embed far apart.  Falls back to a no-op
    (scores all-zero) if PyTorch is unavailable."""

    representation = "intensity_pca"
    needs_training = True

    def __init__(self, emb_dim=64, hidden=128, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 margin=1.0, epochs=80, batch_size=256, neg_per_pos=5, patience=15, seed=0):
        self.name = "M5_Contrastive"
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.margin = margin
        self.epochs = epochs
        self.batch_size = batch_size
        self.neg_per_pos = neg_per_pos
        self.patience = patience
        self.seed = seed
        self.net = None

    def _make_pairs(self, n, patient_ids, rng):
        pid = np.asarray(patient_ids)
        ia, ib, lab = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                ia.append(i); ib.append(j); lab.append(1 if pid[i] == pid[j] else 0)
        ia, ib, lab = np.array(ia), np.array(ib), np.array(lab)
        pos = np.where(lab == 1)[0]
        neg = np.where(lab == 0)[0]
        n_neg = min(len(neg), int(round(self.neg_per_pos * max(len(pos), 1))))
        if n_neg < len(neg):
            neg = rng.choice(neg, size=n_neg, replace=False)
        sel = np.concatenate([pos, neg])
        rng.shuffle(sel)
        return ia[sel], ib[sel], lab[sel].astype(np.float32)

    def fit(self, X_train, patient_ids_train):
        if not _HAVE_TORCH:
            warnings.warn("PyTorch not available -- ContrastiveEmbedding will return zeros")
            return self
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        n = X_train.shape[0]
        ia, ib, lab = self._make_pairs(n, patient_ids_train, rng)
        # small internal validation split (on pairs) for early stopping
        perm = rng.permutation(len(ia))
        ia, ib, lab = ia[perm], ib[perm], lab[perm]
        n_val = max(1, int(0.2 * len(ia)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xt = torch.tensor(X_train, dtype=torch.float32)
        self.net = _EmbedNet(X_train.shape[1], self.emb_dim, self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def loss_fn(d, y):  # contrastive loss
            return torch.mean(y * d ** 2 + (1 - y) * torch.clamp(self.margin - d, min=0.0) ** 2)

        best_val, best_state, bad = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(tr_idx))
            for s in range(0, len(order), self.batch_size):
                b = tr_idx[order[s:s + self.batch_size]]
                if len(b) < 2:
                    continue
                ea = self.net(Xt[ia[b]])
                eb = self.net(Xt[ib[b]])
                d = torch.norm(ea - eb, dim=1)
                loss = loss_fn(d, torch.tensor(lab[b]))
                opt.zero_grad(); loss.backward(); opt.step()
            # validation
            self.net.eval()
            with torch.no_grad():
                ea = self.net(Xt[ia[val_idx]]); eb = self.net(Xt[ib[val_idx]])
                d = torch.norm(ea - eb, dim=1)
                vloss = loss_fn(d, torch.tensor(lab[val_idx])).item()
            if vloss < best_val - 1e-5:
                best_val, best_state, bad = vloss, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def _embed(self, X):
        if not _HAVE_TORCH or self.net is None:
            return None
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy()

    def score_pairs(self, X, ia, ib):
        E = self._embed(X)
        if E is None:
            return np.zeros(len(ia))
        return -np.linalg.norm(E[ia] - E[ib], axis=1)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

def default_models(weighted_saap_kwargs=None):
    """Return a list of (model, ...) instances with the plan's default hyperparameters."""
    wkw = weighted_saap_kwargs or {}
    return [
        JaccardSAAP(),
        WeightedSAAP(**wkw),
        SimilarityPCA(metric="cosine"),
        RandomForestPairs(),
        ContrastiveEmbedding(),
    ]

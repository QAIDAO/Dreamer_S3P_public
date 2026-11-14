from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from .chemistry import fingerprints

def metrics(y, predicted, decision):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"samples": len(y), "accuracy": float(accuracy_score(y, predicted)), "f1_macro": float(f1_score(y, predicted, average="macro")), "roc_auc": float(roc_auc_score(y, decision)), "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}

class Classifier:
    def __init__(self, backend, seed=42):
        self.backend, self.seed = backend, seed
        self.standard = StandardScaler()
        self.pca = PCA(n_components=3, svd_solver="full", random_state=seed)
        self.angles = MinMaxScaler(feature_range=(0, np.pi), clip=True)
        self.quantum = SVC(kernel="precomputed", C=1.0, random_state=seed)
        self.classic3d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
        self.classic128 = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)

    def transform(self, rows):
        x = fingerprints(rows)
        return self.angles.transform(self.pca.transform(self.standard.transform(x)))

    def fit(self, rows):
        labels = np.asarray([int(r["label"]) for r in rows])
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Training data must contain both binary classes")
        x = fingerprints(rows)
        angles = self.angles.fit_transform(self.pca.fit_transform(self.standard.fit_transform(x)))
        self.training_embedding = self.backend.embed(angles, rows)
        self.quantum.fit(self.training_embedding @ self.training_embedding.T, labels)
        self.classic3d.fit(angles, labels)
        self.classic128.fit(x, labels)
        return self

    def evaluate(self, rows):
        y = np.asarray([int(r["label"]) for r in rows])
        angles = self.transform(rows)
        embed = self.backend.embed(angles, rows)
        inputs = {"quantum_features": (self.quantum, embed @ self.training_embedding.T), "classic3d_rbf": (self.classic3d, angles), "classic128_rbf": (self.classic128, fingerprints(rows))}
        results, predictions = {}, []
        for method, (model, x) in inputs.items():
            predicted, decision = model.predict(x), model.decision_function(x)
            results[method] = metrics(y, predicted, decision)
            for r, p, d in zip(rows, predicted, decision):
                predictions.append({"molecule_id": r["molecule_id"], "method": method, "label": int(r["label"]), "predicted_label": int(p), "decision": float(d)})
        return results, predictions

    def rank(self, rows):
        if not rows:
            return []
        angles = self.transform(rows)
        embeddings = self.backend.embed(angles, rows)
        kernel = embeddings @ self.training_embedding.T
        predicted, decision = self.quantum.predict(kernel), self.quantum.decision_function(kernel)
        scores = 1 / (1 + np.exp(-np.clip(decision, -50, 50)))
        result = [{**r, "predicted_label": int(p), "decision": float(d), "model_score": float(s), "execution_source": self.backend.name, "score_calibrated": False} for r, p, d, s in zip(rows, predicted, decision, scores)]
        return sorted(result, key=lambda r: (-r["model_score"], r["molecule_id"]))

    def save_arrays(self, path):
        """Portable numeric artifact; no executable pickle objects."""
        np.savez_compressed(path, standard_mean=self.standard.mean_, standard_scale=self.standard.scale_, pca_components=self.pca.components_, pca_mean=self.pca.mean_, angle_scale=self.angles.scale_, angle_min=self.angles.min_, training_embeddings=self.training_embedding, svm_support=self.quantum.support_, svm_dual_coef=self.quantum.dual_coef_, svm_intercept=self.quantum.intercept_, svm_classes=self.quantum.classes_)

"""CoVA-TAD: conditional-variance aggregation for tabular anomaly detection."""

from typing import Optional, Union
import numpy as np
import torch
from sklearn.base import BaseEstimator, OutlierMixin


class CoVATabularDetector(BaseEstimator, OutlierMixin):
    """
    Training-free tabular anomaly detector based on conditional predictive
    variance from a frozen TabICL regressor.
    
    Parameters
    ----------
    n_projections : int or None, default=None
        Number of feature projection columns. If None or 'all', loops over ALL D features
        without subsampling.
    projection_strategy : {'evenly_spaced', 'random', 'variance_ranked'}, default='evenly_spaced'
        Rule used when ``n_projections`` is smaller than the number of columns.
        ``variance_ranked`` ranks columns by their variance in the unscaled normal
        reference cohort. The strategy is deterministic given ``random_state``.
    bag_size : int, default=200
        Maximum size of each in-context normal prompt bag.
    max_bags : int or None, default=1
        Maximum number of context bags to evaluate. The default uses one bag for
        bounded inference cost. Set to None to cover the full reference cohort.
    chunk_size : int, default=500
        Inference chunk size for queries.
    random_state : int, default=42
        Seed for shuffling normal context instances into bags.
    device : str, default='cpu'
        Device to execute inference on ('cpu' or 'cuda').
    backbone : {'tabicl', 'tabpfn'}, default='tabicl'
        Foundation-model regression backbone to use for conditional density estimation.
    """

    def __init__(
        self,
        n_projections: Optional[Union[int, str]] = None,
        projection_strategy: str = "evenly_spaced",
        bag_size: int = 200,
        max_bags: Optional[int] = 1,
        chunk_size: int = 500,
        random_state: int = 42,
        device: str = "cpu",
        backbone: str = "tabicl",
    ):
        self.n_projections = n_projections
        self.projection_strategy = projection_strategy
        self.bag_size = bag_size
        self.max_bags = max_bags
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.device = device
        self.backbone = backbone

        self.mean_ = None
        self.std_ = None
        self.bags_ = None
        self.selected_cols_ = None
        self.regressor_ = None
        self.model_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Stores normal instances as in-context reference prompt bags and sets up foundation model.
        Zero gradient updates are performed.
        
        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Normal training instances.
        y : Ignored (unsupervised setting).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
            raise ValueError("X must be a non-empty 2D array with at least two features.")
        if self.bag_size < 1 or self.chunk_size < 1:
            raise ValueError("bag_size and chunk_size must be positive integers.")
        if self.max_bags is not None and self.max_bags < 1:
            raise ValueError("max_bags must be a positive integer or None.")
        if self.n_projections not in (None, "all") and int(self.n_projections) < 1:
            raise ValueError("n_projections must be positive, None, or 'all'.")
        valid_projection_strategies = {"evenly_spaced", "random", "variance_ranked"}
        if self.projection_strategy not in valid_projection_strategies:
            raise ValueError(
                "projection_strategy must be one of "
                f"{sorted(valid_projection_strategies)}."
            )
        valid_backbones = {"tabicl", "tabpfn"}
        if self.backbone not in valid_backbones:
            raise ValueError(
                f"backbone must be one of {sorted(valid_backbones)}; got '{self.backbone}'."
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        N, D = X.shape
        self.n_features_in_ = D

        # Compute scaling statistics strictly on normal context
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True) + 1e-6
        X_norm = (X - self.mean_) / self.std_

        # Full-coverage bagging: shuffle and partition to utilize all available normal samples
        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(N)
        all_bags = [indices[i : i + self.bag_size] for i in range(0, N, self.bag_size)]
        if self.max_bags is not None:
            all_bags = all_bags[: self.max_bags]
        self.bags_ = [X_norm[bag] for bag in all_bags]

        # Select projection columns: use ALL features if None or 'all'
        if self.n_projections is None or self.n_projections == "all":
            self.selected_cols_ = list(range(D))
        else:
            num_cols = min(D, int(self.n_projections))
            if self.projection_strategy == "evenly_spaced":
                # Preserve the released default exactly for benchmark compatibility.
                step = max(1, D // num_cols)
                self.selected_cols_ = [
                    i * step for i in range(num_cols) if i * step < D
                ]
            elif self.projection_strategy == "random":
                self.selected_cols_ = sorted(
                    rng.choice(D, size=num_cols, replace=False).tolist()
                )
            else:
                reference_variance = np.var(X, axis=0)
                ranked = np.argsort(-reference_variance, kind="stable")[:num_cols]
                self.selected_cols_ = sorted(ranked.tolist())

        # Initialize pre-trained foundation model weights
        if self.regressor_ is None:
            if self.backbone == "tabicl":
                from tabicl import TabICLRegressor

                rng = np.random.RandomState(self.random_state)
                self.regressor_ = TabICLRegressor(
                    n_estimators=1, device=self.device, random_state=self.random_state
                )
                self.regressor_.fit(rng.standard_normal((10, 2)), rng.standard_normal(10))
                self.model_ = self.regressor_.model_
                self.model_.eval()
            elif self.backbone == "tabpfn":
                try:
                    from tabpfn import TabPFNRegressor

                    self.regressor_ = TabPFNRegressor(
                        n_estimators=1, device=self.device, random_state=self.random_state
                    )
                except ImportError as e:
                    raise ImportError(
                        "TabPFN is not installed. Please install it via 'pip install tabpfn'."
                    ) from e

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the anomaly score for each instance from conditional predictive variance.
        Higher score indicates greater abnormality.
        
        Parameters
        ----------
        X : np.ndarray of shape (M, D)
            Test instances to score.
            
        Returns
        -------
        scores : np.ndarray of shape (M,)
            Aggregate conditional log-variance score.
        """
        if self.bags_ is None:
            raise ValueError("Detector must be fitted before calling decision_function.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be a 2D array with {self.n_features_in_} features; got shape {X.shape}."
            )
        M, D = X.shape
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        X_norm = (X - self.mean_) / self.std_

        bag_scores = []

        for prompt_X in self.bags_:
            total_uncertainty = np.zeros(M, dtype=np.float32)

            for c in self.selected_cols_:
                feat_mask = np.ones(D, dtype=bool)
                feat_mask[c] = False

                prompt_features = prompt_X[:, feat_mask]
                prompt_targets = prompt_X[:, c]

                col_var = []
                if self.backbone == "tabpfn":
                    # Fit TabPFN on prompt features/targets once per projection column
                    self.regressor_.fit(prompt_features, prompt_targets)

                for start in range(0, M, self.chunk_size):
                    end = min(start + self.chunk_size, M)
                    query_features = X_norm[start:end, feat_mask]

                    if self.backbone == "tabicl":
                        batch_eval = np.vstack([prompt_features, query_features])
                        X_t = torch.tensor(batch_eval, dtype=torch.float32, device=self.device).unsqueeze(0)
                        y_t = torch.tensor(prompt_targets, dtype=torch.float32, device=self.device).unsqueeze(0)

                        with torch.no_grad():
                            out = self.model_.predict_stats(
                                X_t,
                                y_t,
                                output_type="variance",
                                inference_config=self.regressor_.inference_config_,
                            )
                            col_var.append(out.squeeze(0).cpu().numpy())
                    elif self.backbone == "tabpfn":
                        # Predict 15.87th and 84.13th percentiles (~1 std deviation)
                        q = self.regressor_.predict(
                            query_features,
                            output_type="quantiles",
                            quantiles=[0.1587, 0.8413],
                        )
                        q_arr = np.asarray(q)
                        if q_arr.ndim == 2 and q_arr.shape[0] == 2 and q_arr.shape[1] != 2:
                            std_est = (q_arr[1] - q_arr[0]) / 2.0
                        elif q_arr.ndim == 2 and q_arr.shape[1] == 2:
                            std_est = (q_arr[:, 1] - q_arr[:, 0]) / 2.0
                        else:
                            std_est = np.std(q_arr, axis=0)
                        col_var.append(np.square(np.maximum(std_est, 0.0)))

                variance = np.maximum(np.concatenate(col_var), 0.0)
                total_uncertainty += np.log(variance + 1e-4)
            bag_scores.append(total_uncertainty)

        # Average conditional log-variance scores across bags.
        return np.mean(bag_scores, axis=0)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Opposite of decision_function for scikit-learn compatibility.
        Lower score indicates greater abnormality.
        """
        return -self.decision_function(X)

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        Predict if a particular sample is an outlier or not.
        
        Parameters
        ----------
        X : np.ndarray of shape (M, D)
            Test instances.
        contamination : float, default=0.1
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used to define the threshold.
            
        Returns
        -------
        is_inlier : np.ndarray of shape (M,)
            Returns -1 for anomalies/outliers and +1 for normal inliers.
        """
        scores = self.decision_function(X)
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must be strictly between 0 and 1.")
        threshold = np.percentile(scores, 100 * (1.0 - contamination))
        preds = np.ones(len(scores), dtype=int)
        preds[scores > threshold] = -1
        return preds


# Backward-compatible alias used by the original exploratory scripts.
TabICLEpistemicDetector = CoVATabularDetector

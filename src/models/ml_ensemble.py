"""
CopIA — Ensemble de Machine Learning
XGBoost + LightGBM + Random Forest com stacking e calibração de probabilidades.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    import lightgbm as lgb
    XGB_AVAILABLE = True
    LGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    LGB_AVAILABLE = False
    logger.warning("XGBoost/LightGBM não instalados. Use: uv sync")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class CopIAEnsemble:
    """
    Ensemble de modelos para predição de resultados de futebol.

    Pipeline:
    1. XGBoost + LightGBM + Random Forest como estimadores base
    2. Logistic Regression como meta-learner (stacking)
    3. Calibração Platt Scaling para probabilidades confiáveis
    4. SHAP para explicabilidade de cada predição
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.calibrated_model = None
        self.feature_columns: list[str] = []
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def _build_estimators(self) -> list:
        """Constrói lista de estimadores base."""
        estimators = []

        if XGB_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
            estimators.append(("xgboost", xgb_model))

        if LGB_AVAILABLE:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            estimators.append(("lightgbm", lgb_model))

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
        )
        estimators.append(("random_forest", rf_model))

        return estimators

    def build(self) -> "CopIAEnsemble":
        """Constrói o pipeline completo de ensemble."""
        estimators = self._build_estimators()

        meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class="multinomial",
            random_state=self.random_state,
        )

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            passthrough=False,  # Usa apenas predições dos base models
            n_jobs=-1,
        )

        # Calibração com Isotonic Regression (melhor para probabilidades)
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.model,
            method="isotonic",
            cv=3,
        )

        logger.info(
            f"Ensemble construído: {[name for name, _ in estimators]} → "
            f"LogisticRegression (stacking) → Isotonic Calibration"
        )
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: list[str] | None = None,
    ) -> "CopIAEnsemble":
        """
        Treina o ensemble.

        Args:
            X: Features de treino
            y: Target (0=derrota, 1=empate, 2=vitória)
            feature_columns: Colunas numéricas a usar (auto-detecta se None)
        """
        if feature_columns is None:
            feature_columns = [c for c in X.columns if X[c].dtype in ["float64", "int64"]]

        self.feature_columns = feature_columns
        X_train = X[feature_columns].fillna(0)

        logger.info(
            f"Treinando ensemble: {len(X_train):,} amostras × "
            f"{len(feature_columns)} features"
        )

        self.calibrated_model.fit(X_train, y)
        self.is_fitted = True

        logger.success("Ensemble treinado com sucesso!")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidades calibradas para [derrota, empate, vitória].
        Shape: (n_samples, 3)
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Execute .fit() primeiro.")
        X_pred = X[self.feature_columns].fillna(0)
        return self.calibrated_model.predict_proba(X_pred)

    def predict_match(
        self,
        features: dict,
    ) -> dict:
        """
        Prediz o resultado de um único confronto.

        Args:
            features: Dict de features do confronto (output de build_matchup_features)

        Returns:
            Dict com probabilidades e placar esperado
        """
        df = pd.DataFrame([features])
        proba = self.predict_proba(df)[0]

        return {
            "p_loss": float(proba[0]),
            "p_draw": float(proba[1]),
            "p_win": float(proba[2]),
            "predicted_winner": (
                features.get("team_a", "Team A") if proba[2] > proba[0]
                else (features.get("team_b", "Team B") if proba[0] > proba[2]
                      else "Empate")
            ),
            "confidence": float(max(proba)),
        }

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
    ) -> dict[str, float]:
        """
        Avalia o modelo com validação cruzada temporal.
        Retorna métricas de qualidade.
        """
        X_eval = X[self.feature_columns].fillna(0)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=False)

        # Accuracy
        acc_scores = cross_val_score(
            self.calibrated_model, X_eval, y, cv=cv, scoring="accuracy"
        )

        # Log Loss
        logloss_scores = cross_val_score(
            self.calibrated_model, X_eval, y, cv=cv, scoring="neg_log_loss"
        )

        metrics = {
            "accuracy_mean": float(acc_scores.mean()),
            "accuracy_std": float(acc_scores.std()),
            "log_loss_mean": float(-logloss_scores.mean()),
            "log_loss_std": float(logloss_scores.std()),
        }

        logger.info("Métricas de avaliação:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    def get_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 100,
    ):
        """
        Calcula SHAP values para explicabilidade.
        Requer que shap esteja instalado.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap não instalado. Execute: uv add shap")

        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado.")

        X_shap = X[self.feature_columns].fillna(0).head(max_samples)

        # Usa TreeExplainer para modelos baseados em árvores
        # Para o ensemble calibrado, usa o estimador interno
        try:
            base_model = self.calibrated_model.estimator
            explainer = shap.TreeExplainer(base_model.final_estimator_)
            # Transforma X para o espaço de predições dos base models
            X_transformed = base_model.transform(X_shap)
            shap_values = explainer(X_transformed)
        except Exception:
            # Fallback: KernelExplainer (mais lento, mas funciona com qualquer modelo)
            explainer = shap.KernelExplainer(
                self.calibrated_model.predict_proba,
                shap.sample(X_shap, 50),
            )
            shap_values = explainer.shap_values(X_shap)

        return shap_values, X_shap

    def save(self, path: Path | None = None) -> Path:
        """Salva o modelo treinado."""
        import joblib
        save_path = path or (MODELS_DIR / "ensemble_model.joblib")
        joblib.dump({
            "model": self.calibrated_model,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
        }, save_path)
        logger.success(f"Modelo salvo: {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Path) -> "CopIAEnsemble":
        """Carrega modelo previamente treinado."""
        import joblib
        data = joblib.load(path)
        instance = cls()
        instance.calibrated_model = data["model"]
        instance.feature_columns = data["feature_columns"]
        instance.is_fitted = data["is_fitted"]
        logger.success(f"Modelo carregado: {path}")
        return instance

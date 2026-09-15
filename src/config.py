"""
集中管理原本散落在 notebook 各處的門檻值與參數。

重構重點：
    原本的 threshold（成交量門檻、當沖比例門檻、振幅門檻…）直接寫死在
    label 計算與特徵工程的程式碼裡，改一個數字要去好幾個地方找。
    抽出來統一管理，之後要做「不同市況重新校準門檻」的實驗會方便很多。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelConfig:
    """定義「是否為當沖股」這個 label 的業務規則。"""

    # 只考慮成交量 > 此門檻的股票，避免流動性太差的股票干擾模型
    min_volume: int = 10_000

    # 每日依「當沖比例」排序，取前 N 檔視為當沖熱門股（正樣本）
    top_n_per_day: int = 30


@dataclass(frozen=True)
class FeatureConfig:
    """衍生特徵（額外的二元旗標）門檻。"""

    high_amplitude_pct: float = 5.0      # 振幅(%) >= 5 視為高振幅
    high_volume_change_pct: float = 10.0  # 成交量變動(%) >= 10 視為成交量變動大
    high_day_trade_ratio: float = 0.3     # 當沖比例 >= 0.3 視為高當沖比率


@dataclass(frozen=True)
class ModelConfig:
    """滯後特徵天數、切分比例、分類門檻。"""

    n_lags: int = 3
    validation_ratio: float = 0.2
    predict_threshold: float = 0.5


@dataclass(frozen=True)
class LightGBMConfig:
    """LightGBM 訓練參數，對應原本寫死在 train() 裡的 params dict。"""

    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    max_depth: int = 6
    min_data_in_leaf: int = 50
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    num_threads: int = 4

    def to_lgb_params(self, scale_pos_weight: float) -> dict:
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "verbose": -1,
            "num_threads": self.num_threads,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": 1e-3,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "scale_pos_weight": scale_pos_weight,
        }


LABEL_CONFIG = LabelConfig()
FEATURE_CONFIG = FeatureConfig()
MODEL_CONFIG = ModelConfig()
LGB_CONFIG = LightGBMConfig()

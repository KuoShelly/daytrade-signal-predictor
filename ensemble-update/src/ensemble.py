"""
模型融合：Average 與 Time-aware Stacking（重構後新增）
========================================================

兩種融合方式都把 LightGBM、Logistic Regression、Random Forest 當作成員模型，
介面與 model_comparison.py 的其他候選模型相同：fit(X, y, dates, feature_names) / predict_proba(X)。

AverageEnsemble
---------------
三個模型的預測機率直接取平均（等權重）。
三個成員都做了等價的類別不平衡處理（class_weight="balanced" / scale_pos_weight = 負/正），
輸出的機率尺度相近，等權重平均不會被某一個模型的分數尺度主導。
優點：不需要額外訓練、不會過度擬合、沒有額外的洩漏風險。

StackingEnsemble（時間序列版）
-----------------------------
一般的 stacking 用隨機 K-fold 產生 out-of-fold（OOF）預測來訓練 meta-model，
但在時間序列上，隨機 K-fold 會讓成員模型用「未來」的資料去預測「過去」，
meta-model 學到的權重會被高估過的成員分數誤導。

這裡改用 expanding-window 的時間切分：
    把訓練期間的日期依序切成 n_folds + 1 段，
    第 k 折：用第 0..k-1 段訓練成員模型，預測第 k 段。
這樣每一筆 OOF 預測都只用到「它之前」的資料，跟上線時的情境一致。
meta-model 是一個 Logistic Regression，輸入是三個成員機率的 logit。
最後成員模型會在「完整訓練期間」重新訓練，用來對驗證集產生輸入。

代價：第一段資料沒有 OOF 預測（沒有更早的資料可訓練），meta-model 的訓練資料
只有訓練期間的後 n_folds/(n_folds+1)；成員模型總共要多訓練 n_folds 次。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def time_ordered_folds(dates: np.ndarray, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window 切分，回傳 [(fit_mask, predict_mask), ...]。
    保證每一折 fit_mask 裡的所有日期都早於 predict_mask 裡的所有日期。
    """
    unique_dates = np.sort(np.unique(dates))
    if len(unique_dates) < n_folds + 1:
        raise ValueError(f"訓練期間只有 {len(unique_dates)} 天，不足以切成 {n_folds + 1} 段")

    blocks = np.array_split(unique_dates, n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        fit_mask = dates < blocks[k][0]
        predict_mask = np.isin(dates, blocks[k])
        folds.append((fit_mask, predict_mask))
    return folds


class AverageEnsemble:
    def __init__(self, member_factories: Dict[str, Callable[[], object]]):
        self.member_factories = member_factories
        self.members: Dict[str, object] = {}

    def fit(self, X, y, dates, feature_names):
        self.members = {
            name: factory().fit(X, y, dates, feature_names)
            for name, factory in self.member_factories.items()
        }
        return self

    def member_predictions(self, X) -> Dict[str, np.ndarray]:
        return {name: m.predict_proba(X) for name, m in self.members.items()}

    def predict_proba(self, X):
        return np.mean(list(self.member_predictions(X).values()), axis=0)


class StackingEnsemble:
    def __init__(
        self,
        member_factories: Dict[str, Callable[[], object]],
        n_folds: int = 4,
        random_state: int = 42,
    ):
        self.member_factories = member_factories
        self.n_folds = n_folds
        self.random_state = random_state
        self.members: Dict[str, object] = {}
        self.meta_model: LogisticRegression | None = None

    def _out_of_fold_predictions(self, X, y, dates, feature_names):
        names = list(self.member_factories)
        oof = np.full((len(y), len(names)), np.nan)
        has_oof = np.zeros(len(y), dtype=bool)

        for fit_mask, predict_mask in time_ordered_folds(dates, self.n_folds):
            for j, name in enumerate(names):
                member = self.member_factories[name]().fit(
                    X[fit_mask], y[fit_mask], dates[fit_mask], feature_names
                )
                oof[predict_mask, j] = member.predict_proba(X[predict_mask])
            has_oof |= predict_mask
        return oof[has_oof], y[has_oof]

    def fit(self, X, y, dates, feature_names):
        oof, y_oof = self._out_of_fold_predictions(X, y, dates, feature_names)

        self.meta_model = LogisticRegression(class_weight="balanced", random_state=self.random_state)
        self.meta_model.fit(_logit(oof), y_oof)

        # 成員模型用完整訓練期間重新訓練，給驗證/上線時使用
        self.members = {
            name: factory().fit(X, y, dates, feature_names)
            for name, factory in self.member_factories.items()
        }
        return self

    def predict_proba(self, X):
        member_scores = np.column_stack([m.predict_proba(X) for m in self.members.values()])
        return self.meta_model.predict_proba(_logit(member_scores))[:, 1]

    @property
    def meta_weights(self) -> Dict[str, float]:
        """meta-model 對每個成員 logit 的係數，可以看出 stacking 最後比較信任哪個模型。"""
        return dict(zip(self.member_factories, self.meta_model.coef_[0]))

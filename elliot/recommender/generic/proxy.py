from typing import Iterable, Optional, Tuple
import ntpath
import numpy as np
import pandas as pd
import torch

from elliot.recommender.base_recommender import BaseRecommender
from elliot.utils.registry import model_registry


@model_registry.register()
class ProxyRecommender(BaseRecommender):
    path: str = ""
    id_space: str = "public"
    deduplicate: bool = True
    filter_seen: bool = True
    strict: bool = False
    model_name: Optional[str] = None

    def __init__(self, params, interactions, seed, *args, **kwargs):
        """
        Create a Proxy recommender to evaluate already generated recommendations.
        :param name: data loader object
        :param path: path to the directory rec. results
        :param args: parameters
        """
        super().__init__(params, interactions, seed, *args, **kwargs)

        self._reader_config = params.meta.rec_reader

        self._public_user_map = getattr(self._interactions, "_u_map")
        self._public_item_map = getattr(self._interactions, "_i_map")
        self._private_user_map = getattr(self._interactions, "_users")
        self._private_item_map = getattr(self._interactions, "_items")
        self._user_id_type, self._item_id_type = self._infer_public_id_types()
        self._train_dict = self._interactions.get_dict(private=True)

        self._seen_items = self._build_seen_items(self._train_dict)
        self._recommendations = self.load_recommendations(self.path)

        self.params_to_save = []

    @property
    def name(self):
        return self.model_name or ntpath.basename(self.path).rsplit(".", 1)[0]

    @property
    def name_param(self):
        return ""

    def train_step(self, batch, *args):
        return 0

    def predict(self, user_indices, item_indices=None):
        num_users = user_indices.shape[0]
        num_items = item_indices.shape[1] if item_indices is not None else self._num_items
        scores = torch.full((num_users, num_items), -torch.inf, dtype=torch.float32)

        user_list = user_indices.tolist()
        item_list = item_indices.tolist() if item_indices is not None else None

        for row, user in enumerate(user_list):
            user_recs = self._recommendations.get(user, {})
            if not user_recs:
                continue
            seen = self._seen_items.get(user) if self.filter_seen else None
            if item_list is None:
                for item, score in user_recs.items():
                    if seen is not None and item in seen:
                        continue
                    scores[row, item] = score
            else:
                for col, item in enumerate(item_list[row]):
                    if item == -1:
                        continue
                    if seen is not None and item in seen:
                        continue
                    score = user_recs.get(item)
                    if score is not None:
                        scores[row, col] = score

        return scores

    def load_recommendations(self, path: str, top_k: Optional[int] = None):
        data = self.reader.read_tabular(
            path=path,
            columns=self._reader_config.column_names(),
            datatypes=self._reader_config.column_dtypes(),
            sep=self._reader_config.sep,
            header=self._reader_config.header,
        )

        data = self._process_tabular_rec(data, top_k=top_k)

        recs = {
            user: dict(zip(group["item_idx"], group["prediction"]))
            for user, group in data.groupby("user_idx", sort=False)
        }

        self.logger.info(
            "Recommendations loaded",
            extra={"context": {"users": len(recs), "rows": len(data)}}
        )

        return recs

    def _process_tabular_rec(self, data: pd.DataFrame, top_k: Optional[int] = None) -> pd.DataFrame:
        column_names = self._reader_config.column_names()
        user_col = column_names[0]
        item_col = column_names[1]
        score_col = column_names[2]

        data, score_col = self._select_columns(data, user_col, item_col, score_col)

        data = data.dropna(subset=["userId", "itemId"]).copy()
        data["userId"] = data["userId"].str.strip()
        data["itemId"] = data["itemId"].str.strip()
        data = data[(data["userId"] != "") & (data["itemId"] != "")]

        data = self._map_ids(data)
        data = self._normalize_predictions(data)

        if data.empty:
            self.logger.warning("No recommendations loaded after normalization")
            return pd.DataFrame()

        data = self._map_internal_ids(data)

        if self.deduplicate:
            before = len(data)
            data = data.sort_values("prediction", ascending=False, kind="mergesort")
            data = data.drop_duplicates(subset=["user_idx", "item_idx"], keep="first")
            dropped = before - len(data)
            if dropped:
                self.logger.info(
                    "Dropped duplicate user-item pairs",
                    extra={"context": {"count": dropped}}
                )

        data = data.sort_values(["user_idx", "prediction"], ascending=[True, False], kind="mergesort")
        if top_k is not None:
            data = data.groupby("user_idx", sort=False).head(top_k)

        return data

    def _infer_public_id_types(self) -> Tuple[type, type]:
        user_type = str
        item_type = str
        if self._public_user_map:
            user_type = type(next(iter(self._public_user_map.keys())))
        elif self._train_dict:
            user_type = type(next(iter(self._train_dict.keys())))

        if self._public_item_map:
            item_type = type(next(iter(self._public_item_map.keys())))
        elif self._train_dict:
            for items in self._train_dict.values():
                if items:
                    if isinstance(items, dict):
                        item_type = type(next(iter(items.keys())))
                    else:
                        item_type = type(next(iter(items)))
                    break

        return user_type, item_type

    def _select_columns(self, data: pd.DataFrame, user_col, item_col, score_col):
        missing = [col for col in (user_col, item_col) if col not in data.columns]
        # If there are missing columns among the ones provided by the user,
        # the reader does not read them
        # if missing:
        #     if self.strict:
        #         raise ValueError(f"Missing columns in recommendation file: {missing}")
        #     self.logger.warning(
        #         "Missing columns, falling back to default indices",
        #         extra={"context": {"missing": missing}}
        #     )
        #     user_col, item_col, score_col = 0, 1, 2
        #     user_col = self._normalize_column_ref(data, user_col)
        #     item_col = self._normalize_column_ref(data, item_col)
        #     score_col = self._normalize_column_ref(data, score_col)
        #     missing = [col for col in (user_col, item_col) if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns in recommendation file: {missing}")

        if score_col not in data.columns:
            if self.strict:
                raise ValueError("Missing score column in recommendation file.")
            score_col = None

        cols = [user_col, item_col] + ([score_col] if score_col is not None else [])
        data = data[cols].copy()
        data.columns = ["userId", "itemId"] + (["prediction"] if score_col is not None else [])
        return data, score_col

    def _map_ids(self, data: pd.DataFrame) -> pd.DataFrame:
        id_space = str(self.id_space).strip().lower()
        if id_space in {"private", "internal"}:
            data = self._map_private_ids(data, "userId", self._private_user_map, "user")
            data = self._map_private_ids(data, "itemId", self._private_item_map, "item")
        else:
            data = self._coerce_public_ids(data, "userId", self._user_id_type, "user")
            data = self._coerce_public_ids(data, "itemId", self._item_id_type, "item")
            data = self._filter_unknown_public_ids(data)
        return data

    def _map_private_ids(self, data: pd.DataFrame, col: str, mapping: Iterable, label: str) -> pd.DataFrame:
        series = pd.to_numeric(data[col], errors="coerce")
        mask = series.notna()
        if mask.sum() < len(data):
            self.logger.warning(
                f"Dropping rows with invalid {label} ids",
                extra={"context": {"count": int(len(data) - mask.sum())}}
            )
        data = data.loc[mask].copy()
        series = series.loc[mask].astype(int)

        if isinstance(mapping, dict):
            data[col] = series.map(mapping)
            before = len(data)
            data = data.dropna(subset=[col])
            dropped = before - len(data)
            if dropped:
                self.logger.warning(
                    f"Dropping rows with unknown {label} ids",
                    extra={"context": {"count": dropped}}
                )
            return data

        mapping_arr = np.asarray(mapping, dtype=object)
        valid = (series >= 0) & (series < len(mapping_arr))
        if valid.sum() < len(series):
            self.logger.warning(
                f"Dropping rows with unknown {label} ids",
                extra={"context": {"count": int(len(series) - valid.sum())}}
            )
        data = data.loc[valid].copy()
        data[col] = mapping_arr[series.loc[valid].to_numpy()]
        return data

    def _coerce_public_ids(self, data: pd.DataFrame, col: str, target_type: type, label: str) -> pd.DataFrame:
        if target_type in (int, np.int32, np.int64) or np.issubdtype(target_type, np.integer):
            series = pd.to_numeric(data[col], errors="coerce")
            mask = series.notna()
            if mask.sum() < len(data):
                self.logger.warning(
                    f"Dropping rows with invalid {label} ids",
                    extra={"context": {"count": int(len(data) - mask.sum())}}
                )
            data = data.loc[mask].copy()
            data[col] = series.loc[mask].astype(int)
            return data
        if target_type in (float, np.float32, np.float64) or np.issubdtype(target_type, np.floating):
            series = pd.to_numeric(data[col], errors="coerce")
            mask = series.notna()
            if mask.sum() < len(data):
                self.logger.warning(
                    f"Dropping rows with invalid {label} ids",
                    extra={"context": {"count": int(len(data) - mask.sum())}}
                )
            data = data.loc[mask].copy()
            data[col] = series.loc[mask].astype(float)
            return data

        data[col] = data[col].astype(str)
        return data

    def _filter_unknown_public_ids(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._public_user_map:
            valid_users = data["userId"].isin(self._public_user_map)
            if valid_users.sum() < len(data):
                self.logger.warning(
                    "Dropping rows with unknown users",
                    extra={"context": {"count": int(len(data) - valid_users.sum())}}
                )
            data = data.loc[valid_users].copy()
        if self._public_item_map:
            valid_items = data["itemId"].isin(self._public_item_map)
            if valid_items.sum() < len(data):
                self.logger.warning(
                    "Dropping rows with unknown items",
                    extra={"context": {"count": int(len(data) - valid_items.sum())}}
                )
            data = data.loc[valid_items].copy()
        return data

    def _normalize_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        if "prediction" not in data.columns:
            data["prediction"] = -data.groupby("userId", sort=False).cumcount().astype(float)
            return data

        series = pd.to_numeric(data["prediction"], errors="coerce")
        mask = series.notna()
        if mask.sum() < len(data):
            self.logger.warning(
                "Dropping rows with invalid prediction values",
                extra={"context": {"count": int(len(data) - mask.sum())}}
            )
        data = data.loc[mask].copy()
        data["prediction"] = series.loc[mask].astype(float)
        return data

    def _map_internal_ids(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._public_user_map:
            data["user_idx"] = data["userId"].map(self._public_user_map)
        else:
            data["user_idx"] = pd.to_numeric(data["userId"], errors="coerce")
        if self._public_item_map:
            data["item_idx"] = data["itemId"].map(self._public_item_map)
        else:
            data["item_idx"] = pd.to_numeric(data["itemId"], errors="coerce")

        before = len(data)
        data = data.dropna(subset=["user_idx", "item_idx"]).copy()
        dropped = before - len(data)
        if dropped:
            self.logger.warning(
                "Dropping rows with unmapped ids",
                extra={"context": {"count": dropped}}
            )

        data["user_idx"] = data["user_idx"].astype(int)
        data["item_idx"] = data["item_idx"].astype(int)
        return data

    @staticmethod
    def _build_seen_items(train_dict: Optional[dict]) -> dict:
        seen = {}
        if not train_dict:
            return seen
        for user, items in train_dict.items():
            if isinstance(items, dict):
                seen[user] = set(items.keys())
            else:
                seen[user] = set(items)
        return seen

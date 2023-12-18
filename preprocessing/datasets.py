import numpy as np
import os
import pandas as pd
import pickle
import json
import random
import torch
from tqdm import tqdm
from typing import Optional, Dict, List, Any, Tuple
from preprocessing.utils import format_query

QueryID = str


class Dataset:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.name = None
        self.seed = seed
        self._set_seeds()

    def _set_seeds(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def get_name(self):
        return self.name

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data


class EntityContextQueryDataset(Dataset):
    def __init__(
        self,
        entities_path: Optional[str] = None,
        contexts_path: Optional[str] = None,
        queries_path: Optional[str] = None,
        # save_dir: str = None,
        max_entities: Optional[int] = None,
        max_contexts: Optional[int] = None,
        seed: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        super().__init__(seed)
        self.name = None
        # if save_dir is None and (entities_path is None or contexts_path is None or queries_path is None):
        #     raise ValueError(
        #         "DatasetBuilder expects either save_dir is not None (and resulting save paths for entities, queries, and contexts will be inferred) OR save_entities_path, save_queries_path, and save_queries_path are ALL specified explicitly."
        #     )
        self.entities_path = entities_path
        self.contexts_path = contexts_path
        self.queries_path = queries_path
        # self.entities_path = (
        #     entities_path if entities_path is not None else os.path.join(save_dir, "entities.json")
        # )
        # self.contexts_path = (
        #     contexts_path if contexts_path is not None else os.path.join(save_dir, "contexts.json")
        # )
        # self.queries_path = (
        #     queries_path if queries_path is not None else os.path.join(save_dir, "queries.json")
        # )

        self.max_entities = max_entities
        self.max_contexts = max_contexts

        self.overwrite = overwrite

    def _save_to_json(self, obj, path) -> None:
        with open(path, "w") as f:
            json.dump(obj, f)
        return path

    def get_contexts_per_query_entity_df(self):
        return pd.DataFrame(
            [
                (q_id, query_form, entity, self.contexts)
                for q_id, query_forms in self.qid_to_queries.items()
                for query_form in query_forms
                for entity in self.entities
            ],
            columns=["q_id", "query_form", "entity", "contexts"],
        )

    def get_entities_per_query_context_df(self):
        return pd.DataFrame(
            [
                (q_id, query_form, context, self.entities)
                for q_id, query_forms in self.qid_to_queries.items()
                for query_form in query_forms
                for context in self.contexts
            ],
            columns=["q_id", "query_form", "context", "entities"],
        )

    def get_df_for_scoring(self):
        return pd.DataFrame(
            [
                (q_id, query_form, entity, context, entity in context)
                for q_id, query_form_to_qec_dict in self.qid_to_query_entity_context_dict.items()
                for (query_form, q_e_c_tuples) in tqdm(query_form_to_qec_dict.items())
                for (entity, context) in q_e_c_tuples
            ],
            columns=["q_id", "query_form", "entity", "context", "context_is_relevant"],
        )

    def construct_query_entity_context_dict(self) -> Dict[QueryID, Dict[str, List[Tuple[str, str, str]]]]:
        """
        Returns a dict of form
        {
            "qid": {
                "query": [
                    (entity, context),
                    ...
                ]
            }
        }
        """
        qid_to_query_entity_context_dict = dict()
        for q_id, query_forms in self.qid_to_queries.items():
            entity_context_tuples = []
            qid_to_query_entity_context_dict[q_id] = dict()
            for query_form in query_forms:
                for entity in self.entities:
                    for context in self.contexts:
                        entity_context_tuples.append((entity, context))
                qid_to_query_entity_context_dict[q_id][query_form] = entity_context_tuples

        return qid_to_query_entity_context_dict

    def build_entities_dataset(self) -> Tuple[List[str], os.PathLike]:
        """
        Returns a tuple containing:
            (1) a list of the entities for a task, represented as a list of strings, and
            (2) a path of a json of the above.

        Example:
        ["China", "USA", "Suriname"]
        """
        raise NotImplementedError("This needs to be implemented in a concrete subclass for the task of interest.")

    def build_contexts_dataset(self) -> Tuple[List[str], os.PathLike]:
        """
        Returns a tuple containing:
            (1) a list of the contexts for a task, represented as a list of strings, and
            (2) a path of a json of the above.

        Example:
        [
            "The capital of China is Kielecki.\n",
            "The capital of USA is Kielecki.\n",
            "The capital of Suriname is Kielecki.\n",
        ]
        """
        raise NotImplementedError("This needs to be implemented in a concrete subclass for the task of interest.")

    def build_queries_dataset(self) -> Tuple[Dict[QueryID, List[str]], os.PathLike]:
        """

        Returns a tuple containing:
            (1) a dict from a query to a list of strings, each of which represents a different formulation of that query, and
            (2) a path of a json of the above.

        Example:
        {
            "qid_1": [
                "Q: What is the capital of {}?\nA:",
                "The capital of {} is",
            ],
        }
        """
        raise NotImplementedError("This needs to be implemented in a concrete subclass for the task of interest.")

    def load_or_build_entities_contexts_and_queries(
        self,
        entities_path: Optional[str] = None,
        contexts_path: Optional[str] = None,
        queries_path: Optional[str] = None,
    ):
        try:
            if self.overwrite:
                raise ValueError("User determined to overwrite existing datasets.")
            self.entities: List[str] = load_dataset_from_path(entities_path)
            self.contexts: List[str] = load_dataset_from_path(contexts_path)
            self.qid_to_queries: Dict[QueryID, List[str]] = load_dataset_from_path(queries_path)
        except OSError:
            print(
                f"Failed to load entities, contexts, and queries from paths {entities_path}, {contexts_path}, and {queries_path}.\nManually reconstructing dataset and saving to aforementioned paths."
            )
        except ValueError:
            print(
                f"Overwriting datasets (if they already exist) at {entities_path}, {contexts_path}, and {queries_path}."
            )
        finally:
            self.entities = self.build_entities_dataset()
            self.contexts = self.build_contexts_dataset()
            self.qid_to_queries = self.build_queries_dataset()


class CountryCapital(EntityContextQueryDataset):
    def __init__(
        self,
        entities_path: Optional[str] = None,
        queries_path: Optional[str] = None,
        contexts_path: Optional[str] = None,
        # save_dir: str = None,
        max_entities: int = None,
        max_contexts: int = None,
        cap_per_type: bool = False,
        seed: Optional[int] = None,
        raw_country_capitals_path: Optional[str] = "../data/CountryCapital/real-fake-country-capital.csv",
    ) -> None:
        super().__init__(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
            max_entities=max_entities,
            max_contexts=max_contexts,
            seed=seed,
            # save_dir=save_dir,
        )
        self.raw_country_capitals_path = raw_country_capitals_path
        self.name = "CountryCapital"
        self.cap_per_type = cap_per_type
        self.load_or_build_entities_contexts_and_queries(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
        )
        self.qid_to_query_entity_context_dict = self.construct_query_entity_context_dict()
        self.df_for_scoring = self.get_df_for_scoring()

    def build_entities_dataset(self) -> List[str]:
        """
        Returns a tuple containing:
            (1) a list of the entities for a task, represented as a list of strings, and
            (2) a path of a json of the above.

        Example:
        ["China", "USA", "Suriname"]
        """
        country_capitals: pd.DataFrame = load_dataset_from_path(self.raw_country_capitals_path)

        if self.max_entities is not None:
            if self.cap_per_type:
                entity_types = country_capitals["type"].unique()
                country_capitals = country_capitals.groupby("type").sample(n=self.max_entities / len(entity_types))
                entities = country_capitals["country"].tolist()
            else:
                entities: List[str] = country_capitals["country"].sample(self.max_entities).tolist()
        else:
            entities: List[str] = country_capitals["country"].tolist()

        if self.entities_path is not None:
            self._save_to_json(entities, self.entities_path)
        else:
            print(f"WARNING: No path provided, so will not try to save entities to path {self.entities_path}.")

        return entities

    def build_contexts_dataset(self) -> List[str]:
        """
        Returns a tuple containing:
            (1) a list of the contexts for a task, represented as a list of strings, and
            (2) a path of a json of the above.

        Example:
        [
            "The capital of China is Kielecki.\n",
            "The capital of USA is Kielecki.\n",
            "The capital of Suriname is Kielecki.\n",
        ]
        """
        country_capitals: pd.DataFrame = load_dataset_from_path(self.raw_country_capitals_path)
        contexts: List[str] = []
        for country in country_capitals["country"]:
            if country in self.entities:
                for capital in country_capitals["capital"]:
                    contexts.append(f"The capital of {country} is {capital}.\n")

        if self.max_contexts is not None:
            contexts = random.sample(contexts, self.max_contexts)

        if self.contexts_path is not None:
            self._save_to_json(contexts, self.contexts_path)
        else:
            print(f"WARNING: No path provided, so will not try to save contexts to path {self.contexts_path}.")

        return contexts

    def build_queries_dataset(self) -> Dict[QueryID, List[str]]:
        """

        Returns a tuple containing:
            (1) a dict from a query to a list of strings, each of which represents a different formulation of that query, and
            (2) a path of a json of the above.

        Example:
        {
            "qid_1": [
                "Q: What is the capital of {}?\nA:",
                "The capital of {} is",
            ],
        }
        """
        query_id_to_queries: Dict[QueryID, List[str]] = {
            "capital_of": [
                "Q: What is the capital of {}?\nA:",
                "The capital of {} is",
            ],
        }

        if self.queries_path is not None:
            self._save_to_json(query_id_to_queries, self.queries_path)
        else:
            print(f"WARNING: No path provided, so will not try to save queries to path {self.queries_path}.")

        return query_id_to_queries


def load_dataset_from_path(path: str, **kwargs):
    """
    Loads a dataset from the path.
    """
    supported_filetypes = {".pickle", ".pt", ".csv", ".tsv", ".json"}
    _, path_suffix = os.path.splitext(path)

    if path_suffix not in supported_filetypes:
        raise ValueError(
            f"load_dataset_from_path currently only loads files of type {supported_filetypes}. Instead received a file with path suffix {path_suffix}."
        )
    else:
        if path_suffix == ".pickle":
            try:
                return pd.read_pickle(path)
            except FileNotFoundError as e:
                print(f"WARNING: unable to read pickle with pandas, instead just loading. Full error: {e}")
                with open(path, "rb") as f:
                    return pickle.load(f)
        elif path_suffix == ".pt":
            return torch.load(path)
        elif path_suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        elif path_suffix == ".tsv":
            return pd.read_csv(path, sep="\t", **kwargs)
        elif path_suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)

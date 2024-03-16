import numpy as np
import os
import pandas as pd
import pickle
import itertools
import json
import random
import sys

# import torch
from tqdm import tqdm
from typing import Optional, Dict, List, Any, Tuple
from preprocessing.utils import format_query, random_sample, top_entity_namesake_degree, top_entity_uri_degree

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
        answers_path: Optional[str] = None,
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
        self.answers_path = answers_path
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
            json.dump(obj, f, ensure_ascii=False, indent=4)
        return path

    def get_contexts_per_query_entity_df(self):
        return pd.DataFrame(
            [
                (q_id, query_form, entity, answer, self.contexts)
                for q_id, query_forms in self.qid_to_queries.items()
                for query_form in query_forms
                for entity, answer in zip(self.entities, self.answers)
            ],
            columns=["q_id", "query_form", "entity", "answer", "contexts"],
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

    def build_entities_and_answers_dataset(self) -> Tuple[List[Tuple[str]], List[str]]:
        """
        Returns a tuple containing:
            (1) a list of the entities for a task, represented as a list of tuples of strings, and
            (2) a list of the corresponding answer for each entities (for the query defining this dataset instance)

        Example (if the task is about country capitals):
        (
            [
                ("China",),
                ("USA",),
                ("Suriname",),
            ],
            [
                "Beijing",
                "Washington DC",
                "Paramaribo",
            ]
        )
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
        answers_path: Optional[str] = None,
    ):
        try:
            # TODO: fix bug, this try block is always being overwritten by the finally block
            if self.overwrite:
                raise ValueError("User determined to overwrite existing datasets.")
            self.entities: List[Tuple[str]] = load_dataset_from_path(entities_path)
            self.answers: List[Tuple[str]] = load_dataset_from_path(answers_path)
            self.contexts: List[str] = load_dataset_from_path(contexts_path)
            self.qid_to_queries: Dict[QueryID, List[str]] = load_dataset_from_path(queries_path)
        except (OSError, TypeError):
            print(
                f"Failed to load entities, answers, contexts, and queries from paths {entities_path}, {answers_path}, {contexts_path}, and {queries_path}.\nManually reconstructing dataset and saving to aforementioned paths."
            )
        except ValueError:
            print(
                f"Overwriting datasets (if they already exist) at {entities_path}, {answers_path}, {contexts_path}, and {queries_path}."
            )
        finally:
            self.entities, self.answers = self.build_entities_and_answers_dataset()
            self.contexts = self.build_contexts_dataset()
            self.qid_to_queries = self.build_queries_dataset()


class CountryCapital(EntityContextQueryDataset):
    def __init__(
        self,
        entities_path: Optional[str] = None,
        queries_path: Optional[str] = None,
        contexts_path: Optional[str] = None,
        answers_path: Optional[str] = None,
        # save_dir: str = None,
        max_entities: int = None,
        max_contexts: int = None,
        cap_per_type: bool = False,
        ablate_out_relevant_contexts: bool = False,
        uniform_contexts: bool = False,
        deduplicate_entities: bool = False,
        entity_selection_func_name: str = "random_sample",
        seed: Optional[int] = None,
        raw_data_path: Optional[str] = "../data/CountryCapital/real-fake-country-capital.csv",
        overwrite: bool = False,
    ) -> None:
        super().__init__(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
            answers_path=answers_path,
            max_entities=max_entities,
            max_contexts=max_contexts,
            seed=seed,
            overwrite=overwrite,
            # save_dir=save_dir,
        )
        self.raw_data_path = raw_data_path
        self.name = "CountryCapital"
        self.cap_per_type = cap_per_type
        self.ablate_out_relevant_contexts = ablate_out_relevant_contexts
        self.load_or_build_entities_contexts_and_queries(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
        )
        self.qid_to_query_entity_context_dict = self.construct_query_entity_context_dict()
        # self.df_for_scoring = self.get_df_for_scoring()

    def build_entities_and_answers_dataset(self) -> Tuple[List[Tuple[str]], List[str]]:
        """
        Returns a tuple containing:
            (1) a list of the entities for a task, represented as a list of tuples of strings, and
            (2) a list of the corresponding answer for each entities (for the query defining this dataset instance)

        Example (if the task is about country capitals):
        (
            [
                ("China",),
                ("USA",),
                ("Suriname",),
            ],
            [
                "Beijing",
                "Washington DC",
                "Paramaribo",
            ]
        )
        """
        country_capitals: pd.DataFrame = load_dataset_from_path(self.raw_data_path)

        if self.max_entities is not None:
            if self.cap_per_type:
                entity_types = country_capitals["type"].unique()
                country_capitals = country_capitals.groupby("type").sample(n=int(self.max_entities / len(entity_types)))
                entities_and_answers = country_capitals[["country", "capital"]]
            else:
                entities_and_answers: List[str] = country_capitals[["country", "capital"]].sample(self.max_entities)
        else:
            entities_and_answers: List[str] = country_capitals[["country", "capital"]]

        entities = entities_and_answers["country"].tolist()
        entities = [(e,) for e in entities]
        answers = entities_and_answers["capital"].tolist()  # TODO: implement answers for this task

        if self.entities_path is not None:
            self._save_to_json(entities, self.entities_path)
        else:
            print(f"WARNING: No path provided, so will not try to save entities to path {self.entities_path}.")

        return entities, answers

    def build_contexts_dataset(self) -> List[str]:
        """
        Returns a list of the contexts for a task, represented as a list of strings.

        Example:
        [
            "The capital of China is Kielecki.\n",
            "The capital of USA is Kielecki.\n",
            "The capital of Suriname is Kielecki.\n",
        ]
        """
        country_capitals: pd.DataFrame = load_dataset_from_path(self.raw_data_path)
        contexts: List[str] = []
        for country in country_capitals["country"]:
            if (self.ablate_out_relevant_contexts and (country,) not in self.entities) or (
                not self.ablate_out_relevant_contexts and (country,) in self.entities
            ):
                # if self.ablate_out_relevant_contexts ^ ((country,) in self.entities):
                # XOR - if you're ablating out all relevant contexts, then exclude all countries in your entity list.
                #       OR if you're keeping relevant contexts, then include only countries in self.entities.
                # self.entities is a list of single-element tuples
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

        Returns a dict from a query to a list of strings, each of which represents a different formulation of that query.

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


class FriendEnemy(EntityContextQueryDataset):
    def __init__(
        self,
        entities_path: Optional[str] = None,
        queries_path: Optional[str] = None,
        contexts_path: Optional[str] = None,
        answers_path: Optional[str] = None,
        # save_dir: str = None,
        max_entities: int = None,
        max_contexts: int = None,
        cap_per_type: bool = False,
        ablate_out_relevant_contexts: bool = False,
        uniform_contexts: bool = False,
        deduplicate_entities: bool = False,
        entity_selection_func_name: str = "random_sample",
        seed: Optional[int] = None,
        raw_data_path: Optional[str] = "../data/FriendEnemy/raw-friend-enemy.csv",
        overwrite: bool = False,
    ) -> None:
        super().__init__(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
            max_entities=max_entities,
            max_contexts=max_contexts,
            seed=seed,
            overwrite=overwrite,
            # save_dir=save_dir,
        )
        self.raw_data_path = raw_data_path
        self.name = "FriendEnemy"
        self.cap_per_type = cap_per_type
        self.load_or_build_entities_contexts_and_queries(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
        )
        self.qid_to_query_entity_context_dict = self.construct_query_entity_context_dict()
        # self.df_for_scoring = self.get_df_for_scoring()

    def build_entities_and_answers_dataset(self) -> Tuple[List[Tuple[str]], List[str]]:
        """
        Returns a tuple containing:
            (1) a list of the entities for a task, represented as a list of tuples of strings, and
            (2) a list of the corresponding answer for each entities (for the query defining this dataset instance)
        Example:
        (
            [
                ("China", "USA"),
                ("Batman", "Joker"),
            ],
            [
                None,
                None,
            ]
        )

        This case is weird because there are no answers for this task. Maybe this formulation isn't right hm...
        """
        friend_enemy_pairs: pd.DataFrame = load_dataset_from_path(self.raw_data_path)

        if self.max_entities is not None:
            if self.cap_per_type:
                entity_types = friend_enemy_pairs["type"].unique()
                friend_enemy_pairs = friend_enemy_pairs.groupby("type").sample(n=self.max_entities / len(entity_types))
                entities = list(zip(friend_enemy_pairs["ent1"].tolist(), friend_enemy_pairs["ent2"].tolist()))
            else:
                sample = friend_enemy_pairs.sample(self.max_entities)
                entities = list(zip(sample["ent1"].tolist(), sample["ent2"].tolist()))
        else:
            entities: List[str] = list(zip(friend_enemy_pairs["ent1"].tolist(), friend_enemy_pairs["ent2"].tolist()))

        answers = [None] * len(entities)  # TODO: implement answers for this task
        if self.entities_path is not None:
            self._save_to_json(entities, self.entities_path)
        else:
            print(f"WARNING: No path provided, so will not try to save entities to path {self.entities_path}.")

        return entities, answers

    def build_contexts_dataset(self) -> List[str]:
        """
        Returns a list of the contexts for a task, represented as a list of strings, and

        Example:
        [
            "The capital of China is Kielecki.\n",
            "The capital of USA is Kielecki.\n",
            "The capital of Suriname is Kielecki.\n",
        ]
        """
        pos_contexts = [
            "{} loves {}.\n",
            "{} adores {}.\n",
            "{} likes {}.\n",
            # "{} appreciates {}.\n",
            # "{} hugs {}.\n",
            # "{} warmly shakes hands with {}.\n",
            # "{} smiles at {}.\n",
            # "{} kisses {}.\n",
            # "{} trusts {}.\n",
        ]
        neutral_contexts = [
            "{} meets {}.\n",
            # "{} greets {}.\n",
            # "{} passes {}.\n",
            "{} sees {}.\n",
            "{} acknowledges {}.\n",
            # "{} shakes hands with {}.\n",
            # "{} calls {}.\n",
            # "{} forgets about {}.\n",
            # "{} acquaints with {}.\n",
        ]
        neg_contexts = [
            "{} hates {}.\n",
            "{} detests {}.\n",
            "{} dislikes {}.\n",
            # "{} ignores {}.\n",
            # "{} attacks {}.\n",
            # "{} tricks {}.\n",
            # "{} spies on {}.\n",
            # "{} distrusts {}.\n",
            # "{} avoids {}.\n",
        ]
        raw_contexts = pos_contexts + neutral_contexts + neg_contexts

        df: pd.DataFrame = load_dataset_from_path(self.raw_data_path)
        contexts: List[str] = []
        for _, row in df.iterrows():
            if (row["ent1"], row["ent2"]) in self.entities:
                for context in raw_contexts:
                    contexts.append(context.format(row["ent1"], row["ent2"]))

        if self.max_contexts is not None:
            contexts = random.sample(contexts, self.max_contexts)

        if self.contexts_path is not None:
            self._save_to_json(contexts, self.contexts_path)
        else:
            print(f"WARNING: No path provided, so will not try to save contexts to path {self.contexts_path}.")

        return contexts

    def build_queries_dataset(self) -> Dict[QueryID, List[str]]:
        """
        Returns a dict from a query to a list of strings, each of which represents a different formulation of that query, and

        Example:
        {
            "qid_1": [
                "Q: What is the capital of {}?\nA:",
                "The capital of {} is",
            ],
        }
        """
        query_id_to_queries: Dict[QueryID, List[str]] = {
            "friend-enemy": [
                # "Q: Are {} and {} friends or enemies?\nA:",
                # "Q: How friendly are {} and {}?\nA:",
                # "Q: What is the relationship between {} and {}?\nA:",
                # "{} and {} are",
                "The relationship between {} and {} is",
                "Q: On a scale of 1-5, where 1 indicates worst enemies and 5 indicates best friends, how friendly are {} and {}?\nA:",
                "On a scale of 1-5, where 1 indicates worst enemies and 5 indicates best friends, the friendliness level of {} and {} is",
                "Q: On a scale of 1-5, how friendly are {} and {}?\nA:",
                "On a scale of 1-5, the friendliness level of {} and {} is",
            ],
        }

        if self.queries_path is not None:
            self._save_to_json(query_id_to_queries, self.queries_path)
        else:
            print(f"WARNING: No path provided, so will not try to save queries to path {self.queries_path}.")

        return query_id_to_queries


class WorldLeaders(EntityContextQueryDataset):
    def __init__(
        self,
        entities_path: Optional[str] = None,
        queries_path: Optional[str] = None,
        contexts_path: Optional[str] = None,
        # save_dir: str = None,
        max_entities: int = None,
        max_contexts: int = None,
        cap_per_type: bool = False,
        ablate_out_relevant_contexts: bool = False,
        seed: Optional[int] = None,
        raw_data_path: Optional[str] = "../data/WorldLeaders/world-leaders-2001-to-2021.csv",
        overwrite: bool = False,
    ) -> None:
        super().__init__(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
            max_entities=max_entities,
            max_contexts=max_contexts,
            seed=seed,
            overwrite=overwrite,
            # save_dir=save_dir,
        )
        self.raw_data_path = raw_data_path
        self.name = "WorldLeaders"
        self.cap_per_type = cap_per_type
        self.ablate_out_relevant_contexts = ablate_out_relevant_contexts
        self.load_or_build_entities_contexts_and_queries(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
        )
        self.qid_to_query_entity_context_dict = self.construct_query_entity_context_dict()
        # self.df_for_scoring = self.get_df_for_scoring()

    def build_entities_and_answers_dataset(self) -> Tuple[List[Tuple[str]], List[str]]:
        """
        Returns a tuple containing:
            (1) a list of the entities for a task, represented as a list of tuples of strings, and
            (2) a list of the corresponding answer for each entities (for the query defining this dataset instance)

        Example:
        (
            [
                ("Biden",),
                ("Xi Jinping",),
                ("Macron",),
            ],
            [
                "USA",
                "China",
                "France",
            ]
        )
        """
        df: pd.DataFrame = load_dataset_from_path(self.raw_data_path)

        if self.max_entities is not None:
            if self.cap_per_type:
                entity_types = df["type"].unique()
                df = df.groupby("type").sample(n=int(self.max_entities / len(entity_types)))
                entities = df["country"].tolist()
            else:
                entities: List[str] = df["leader"].sample(self.max_entities).tolist()
        else:
            entities: List[str] = df["leader"].tolist()

        entities = [(e,) for e in entities]
        answers = [None] * len(entities)  # TODO: implement answers for this task

        if self.entities_path is not None:
            self._save_to_json(entities, self.entities_path)
        else:
            print(f"WARNING: No path provided, so will not try to save entities to path {self.entities_path}.")

        return entities, answers

    def build_contexts_dataset(self) -> List[str]:
        """
        Returns a list of the contexts for a task, represented as a list of strings.

        Example:
        [
            "Biden is the leader of USA.\n",
            "Biden is the leader of China.\n",
            "Biden is the leader of Suriname.\n",
        ]
        """
        df: pd.DataFrame = load_dataset_from_path(self.raw_data_path)
        contexts: List[str] = []
        for leader in df["leader"]:
            if (self.ablate_out_relevant_contexts and (leader,) not in self.entities) or (
                not self.ablate_out_relevant_contexts and (leader,) in self.entities
            ):
                # if self.ablate_out_relevant_contexts ^ ((country,) in self.entities):
                # XOR - if you're ablating out all relevant contexts, then exclude all countries in your entity list.
                #       OR if you're keeping relevant contexts, then include only countries in self.entities.
                # self.entities is a list of single-element tuples
                for country in df["country"]:
                    contexts.append(f"{leader} is the leader of {country}.\n")

        if self.max_contexts is not None:
            contexts = random.sample(contexts, self.max_contexts)

        if self.contexts_path is not None:
            self._save_to_json(contexts, self.contexts_path)
        else:
            print(f"WARNING: No path provided, so will not try to save contexts to path {self.contexts_path}.")

        return contexts

    def build_queries_dataset(self) -> Dict[QueryID, List[str]]:
        """

        Returns a dict from a query to a list of strings, each of which represents a different formulation of that query.

        Example:
        {
            "qid_1": [
                "Q: What country does {entity} lead?\nA:",
                "{entity} leads"
            ],
        }
        """
        query_id_to_queries: Dict[QueryID, List[str]] = {
            "capital_of": [
                "Q: What country is led by {}?\nA:",
                "{} is the leader of",
            ],
        }

        if self.queries_path is not None:
            self._save_to_json(query_id_to_queries, self.queries_path)
        else:
            print(f"WARNING: No path provided, so will not try to save queries to path {self.queries_path}.")

        return query_id_to_queries


class YagoECQ(EntityContextQueryDataset):
    def __init__(
        self,
        subname: str,
        query_id: str,
        query_types: List[str] = ["closed"],
        entity_types: List[str] = ["entities"],
        context_types: List[str] = ["base"],
        entities_path: Optional[str] = None,
        queries_path: Optional[str] = None,
        contexts_path: Optional[str] = None,
        answers_path: Optional[str] = None,
        # save_dir: str = None,
        max_entities: int = None,
        max_contexts: int = None,
        cap_per_type: bool = False,
        ablate_out_relevant_contexts: bool = False,
        uniform_contexts: bool = False,
        deduplicate_entities: bool = False,
        entity_selection_func_name: str = "random_sample",
        seed: Optional[int] = None,
        raw_data_path: Optional[str] = "../data/YagoECQ/yago_qec.json",
        overwrite: bool = False,
    ) -> None:
        super().__init__(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
            answers_path=answers_path,
            max_entities=max_entities,
            max_contexts=max_contexts,
            seed=seed,
            overwrite=overwrite,
            # save_dir=save_dir,
        )
        self.raw_data_path = raw_data_path
        self.name = "YagoECQ"
        self.subname = subname
        self.query_id = query_id

        self.query_types = query_types
        valid_query_types = {"closed", "open"}
        if not set(self.query_types).issubset(valid_query_types):
            raise ValueError(f"query_types must be subset of {valid_query_types}, instead received {query_types}.")

        self.entity_types = entity_types
        valid_entity_types = {
            "entities",
            "fake_entities",
            "gpt_fake_entities",
            "my_famous_entities",
            "my_fake_entities",
        }
        if not set(self.entity_types).issubset(valid_entity_types):
            raise ValueError(f"entity_types must be subset of {valid_entity_types}, instead received {entity_types}.")

        self.context_types = context_types
        valid_context_types = {"base", "assertive", "super_assertive", "ignore_prior", "negation", "believe_me"}
        if not set(self.context_types).issubset(valid_context_types):
            raise ValueError(
                f"context_types must be subset of {valid_context_types}, instead received {context_types}."
            )

        self.cap_per_type = cap_per_type
        self.ablate_out_relevant_contexts = ablate_out_relevant_contexts
        self.uniform_contexts = uniform_contexts
        self.deduplicate_entities = deduplicate_entities
        self.entity_selection_func = getattr(sys.modules[__name__], entity_selection_func_name)
        self.context_templates: Dict[str, str] = self._extract_context_templates()
        self.load_or_build_entities_contexts_and_queries(
            entities_path=entities_path,
            queries_path=queries_path,
            contexts_path=contexts_path,
            answers_path=answers_path,
        )
        self.qid_to_query_entity_context_dict = self.construct_query_entity_context_dict()

    def _extract_context_templates(self) -> List[str]:
        yago_qec: dict = load_dataset_from_path(self.raw_data_path)[self.query_id]
        return {ct: yago_qec["context_templates"][ct] for ct in self.context_types}

    def _read_entities(self, yago_qec: dict) -> List[str]:
        """Helper to read the eligible entities from yago_qec"""
        # return yago_qec["entities"] # only real entities
        # TODO: remove this hack for handling entity types having different number of entities (we should enforce an equal number of each entity type)
        min_length = min([len(yago_qec[t]) for t in self.entity_types])
        return list(
            itertools.chain.from_iterable([yago_qec[t][:min_length] for t in self.entity_types])
        )  # return entities of all types in self.entity_types

    def build_entities_and_answers_dataset(self) -> Tuple[List[Tuple[str]], List[str]]:
        """
        Returns
            a list of the entities for a task, represented as a list of (single-element) tuples of strings
            AND
            a list of the answers corresponding to each entity (for the query defining this dataset instance)

        Example:
        (
            [
                ("Biden",),
                ("Xi Jinping",),
                ("Macron",),
            ],
            [
                "USA",
                "China",
                "France",
            ]
        )
        """
        yago_qec: dict = load_dataset_from_path(self.raw_data_path)[self.query_id]
        # check if all entity types have the same number of entities
        if not all(len(yago_qec[t]) == len(yago_qec[self.entity_types[0]]) for t in self.entity_types):
            print(
                f"Number of entities in each entity type must be the same, instead received lengths {[len(yago_qec[t]) for t in self.entity_types]} {self.entity_types}. Truncating entity size per type to be {min([len(yago_qec[t]) for t in self.entity_types])}"
            )
            # raise ValueError(
            #     f"Number of entities in each entity type must be the same, instead received lengths {[len(yago_qec[t]) for t in self.entity_types]} {self.entity_types}."
            # )

        entities: List[str] = self._read_entities(yago_qec)
        num_entities_per_type = len(entities) // len(self.entity_types)
        answers: List[str] = yago_qec["answers"][:num_entities_per_type] * len(
            self.entity_types
        )  # TODO: this might be sketchy, since it assumes that

        entities_and_answers: List[Tuple[str, str]] = list(zip(entities, answers))

        if self.deduplicate_entities:
            # Deduplicate rows where the same entity appears multiple times
            entities_and_answers = list(
                pd.DataFrame(entities_and_answers, columns=["entities", "answers"])
                .groupby("entities")
                .agg("sample")
                .itertuples(name=None, index=False)
            )

        if self.max_entities is not None:
            entities_and_answers: List[str] = self.entity_selection_func(
                entities_and_answers, yago_qec, self.max_entities
            )

        entities: List[Tuple[str]] = [(e,) for e, _ in entities_and_answers]
        answers: List[str] = [a for _, a in entities_and_answers]

        if self.entities_path is not None:
            self._save_to_json(entities, self.entities_path)
        else:
            print(f"WARNING: No path provided, so will not try to save entities to path {self.entities_path}.")

        if self.answers_path is not None:
            self._save_to_json(answers, self.answers_path)
        else:
            print(f"WARNING: No path provided, so will not try to save answers to path {self.answers_path}.")

        return entities, answers

    # def build_entities_dataset(self) -> List[Tuple[str]]:
    #     """
    #     Returns a list of the entities for a task, represented as a list of (single-element) tuples of strings

    #     Example:
    #     [
    #         ("Biden",),
    #         ("Xi Jinping",),
    #         ("Macron",),
    #     ]
    #     """
    #     if self.entities_path is None or self.answers_path is None or not os.path.isfile(self.entities_path) or not os.path.isfile(self.answers_path):
    #         entities, _ = self.build_entities_and_answers_dataset()
    #     else:
    #         entities: List[Tuple[str]] = load_dataset_from_path(self.entities_path)
    #     return entities

    # def build_answers_dataset(self) -> List[str]:
    #     """
    #     Returns a list of the entities for a task, represented as a list of (single-element) tuples of strings

    #     Example:
    #     [
    #         ("Biden",),
    #         ("Xi Jinping",),
    #         ("Macron",),
    #     ]
    #     """
    #     if self.entities_path is None or self.answers_path is None or not os.path.isfile(self.entities_path) or not os.path.isfile(self.answers_path):
    #         _, answers = self.build_entities_and_answers_dataset()
    #     else:
    #         answers: List[Tuple[str]] = load_dataset_from_path(self.answers_path)
    #     return answers

    def build_contexts_dataset(self) -> List[str]:
        """
        Returns a list of the contexts for a task, represented as a list of strings.

        Example:
        [
            "Biden is the leader of USA.\n",
            "Biden is the leader of China.\n",
            "Biden is the leader of Suriname.\n",
        ]
        """
        yago_qec: dict = load_dataset_from_path(self.raw_data_path)[self.query_id]
        entities: List[str] = self._read_entities(yago_qec)
        answers: List[str] = yago_qec["answers"]

        context_per_entity_per_answer: List[str] = []
        for entity in entities:
            if (self.ablate_out_relevant_contexts and (entity,) not in self.entities) or (
                not self.ablate_out_relevant_contexts and (entity,) in self.entities
            ):
                # XOR - if you're ablating out all relevant contexts, then exclude all countries in your entity list.
                #       OR if you're keeping relevant contexts, then include only countries in self.entities.
                # self.entities is a list of single-element tuples
                for answer in answers:
                    for ct, context_template in self.context_templates.items():
                        context_per_entity_per_answer.append(
                            (entity, answer, context_template.format(entity=entity, answer=answer), ct)
                        )

        contexts: List[str] = [c for _, _, c, _ in context_per_entity_per_answer]
        if self.max_contexts is not None:
            if self.uniform_contexts:
                contexts = (
                    pd.DataFrame(context_per_entity_per_answer, columns=["entity", "answer", "context", "context_type"])
                    .groupby(["entity", "context_type"])
                    .sample(n=self.max_contexts // len(self.entities) // len(self.context_types))["context"]
                    .tolist()
                )
            else:
                contexts = random.sample(contexts, self.max_contexts)

        if self.contexts_path is not None:
            self._save_to_json(contexts, self.contexts_path)
        else:
            print(f"WARNING: No path provided, so will not try to save contexts to path {self.contexts_path}.")

        return contexts

    def build_queries_dataset(self) -> Dict[QueryID, List[str]]:
        """

        Returns a dict from a query to a list of strings, each of which represents a different formulation of that query.

        Example:
        {
            "qid_1": [
                "Q: What country does {entity} lead?\nA:",
                "{entity} leads"
            ],
        }
        """
        yago_qec: dict = load_dataset_from_path(self.raw_data_path)[self.query_id]
        query_forms = list(itertools.chain.from_iterable([yago_qec["query_forms"][qt] for qt in self.query_types]))
        query_id_to_queries: Dict[QueryID, List[str]] = {
            self.query_id: query_forms,
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

import numpy as np
import torch
import unittest as ut
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from measuring.estimate_probs import (
    estimate_prob_y_given_context_and_entity,
    estimate_prob_x_given_e,
    estimate_prob_next_word_given_x_and_entity,
    estimate_cmi,
    score_model_for_next_word_prob,
    create_position_ids_from_input_ids,
    sharded_score_model,
    estimate_entity_score,
    kl_div,
)


class TestEstimateProbXGivenE(ut.TestCase):
    def test_estimate_prob_x_given_e_uniform(self):
        contexts = [
            "The movie was great.",
            "I loved it.",
            "I hated this.",
        ]
        entity = "entity1"

        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        actual = estimate_prob_x_given_e(entity=entity, contexts=contexts)
        assert np.array_equal(actual, expected)


class TestEstimateProbNextWordGivenXAndEntity(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
        ).to(device)

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name,
            padding_side="left",
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model.config.pad_token_id = cls.model.config.eos_token_id

    @patch("measuring.estimate_probs.score_model_for_next_word_prob")
    def test_estimate_prob_next_word_given_x_and_entity_sans_answer_map_bs1(self, mock_score_model_for_next_word_prob):
        model = type(self).model
        tokenizer = type(self).tokenizer

        mock_outputs = [
            torch.arange(model.config.vocab_size).unsqueeze(0),  # shape: (1, vocab_sz)
            torch.arange(10, model.config.vocab_size + 10).unsqueeze(0),
            torch.ones(model.config.vocab_size).unsqueeze(0),
        ]
        mock_score_model_for_next_word_prob.side_effect = mock_outputs

        contexts = [
            "The movie was great. ",
            "I loved it. ",
            "I hated this. ",
        ]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = "entity1"

        actual = estimate_prob_next_word_given_x_and_entity(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            bs=1,
            answer_map=None,
        )

        expected = torch.nn.functional.softmax(torch.cat(mock_outputs, dim=0), dim=1)

        assert np.allclose(actual, expected)

    @patch("measuring.estimate_probs.score_model_for_next_word_prob")
    def test_estimate_prob_next_word_given_x_and_entity_sans_answer_map_bs_all(
        self, mock_score_model_for_next_word_prob
    ):
        model = type(self).model
        tokenizer = type(self).tokenizer

        mock_outputs = torch.cat(
            [
                torch.arange(model.config.vocab_size).unsqueeze(0),  # shape: (1, vocab_sz)
                torch.arange(10, model.config.vocab_size + 10).unsqueeze(0),
                torch.ones(model.config.vocab_size).unsqueeze(0),
            ],
            dim=0,
        )
        mock_score_model_for_next_word_prob.side_effect = [mock_outputs]

        contexts = [
            "The movie was great. ",
            "I loved it. ",
            "I hated this. ",
        ]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = "entity1"

        actual = estimate_prob_next_word_given_x_and_entity(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            bs=32,
            answer_map=None,
        )

        expected = torch.nn.functional.softmax(mock_outputs, dim=1)

        assert np.allclose(actual, expected)


class TestEstimateProbYGivenXAndEntity(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
        ).to(device)

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name,
            padding_side="left",
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model.config.pad_token_id = cls.model.config.eos_token_id

    @patch("measuring.estimate_probs.estimate_prob_next_word_given_x_and_entity")
    @patch("measuring.estimate_probs.sample_y_given_x_and_entity")
    def test_estimate_prob_y_given_x_and_entity_calls_estimate_prob_if_no_samples(
        self, mock_sample_y_given_x_and_entity, mock_estimate_prob_next_word_given_x_and_entity
    ):
        model = type(self).model
        tokenizer = type(self).tokenizer

        contexts = [
            "The movie was great. ",
            "I loved it. ",
            "I hated this. ",
        ]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = "entity1"

        estimate_prob_y_given_context_and_entity(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            num_samples=None,
            max_output_length=1,
            answer_map=None,
        )

        mock_estimate_prob_next_word_given_x_and_entity.assert_called_once_with(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            answer_map=None,
        )

        mock_sample_y_given_x_and_entity.assert_not_called()

    @patch("measuring.estimate_probs.estimate_prob_next_word_given_x_and_entity")
    @patch("measuring.estimate_probs.sample_y_given_x_and_entity")
    def test_estimate_prob_y_given_x_and_entity_calls_sample_if_num_samples_specified(
        self, mock_sample_y_given_x_and_entity, mock_estimate_prob_next_word_given_x_and_entity
    ):
        model = type(self).model
        tokenizer = type(self).tokenizer

        contexts = [
            "The movie was great. ",
            "I loved it. ",
            "I hated this. ",
        ]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = "entity1"

        estimate_prob_y_given_context_and_entity(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            num_samples=32,
            max_output_length=1,
            answer_map=None,
        )

        mock_sample_y_given_x_and_entity.assert_called_once_with(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            num_samples=32,
            max_output_length=1,
        )

        mock_estimate_prob_next_word_given_x_and_entity.assert_not_called()


class TestCreatePositionIds(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.padding_idx = 23

    def test_create_position_ids_from_input_ids_single_example_with_padding(self):
        padding_idx = type(self).padding_idx
        input_ids = torch.tensor(
            [
                [padding_idx, 1, 4, 2, 19, 28],
            ],
            dtype=torch.long,
        )

        expected_position_ids = torch.tensor(
            [
                [0, 0, 1, 2, 3, 4],
            ],
            dtype=torch.long,
        )

        actual_position_ids = create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=padding_idx)

        assert torch.equal(actual_position_ids, expected_position_ids)

    def test_create_position_ids_from_input_ids_single_example_sans_padding(self):
        padding_idx = type(self).padding_idx
        input_ids = torch.tensor(
            [
                [29, 1, 4, 2, 19, 28],
            ],
            dtype=torch.long,
        )

        expected_position_ids = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        )

        actual_position_ids = create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=padding_idx)

        assert torch.equal(actual_position_ids, expected_position_ids)

    def test_create_position_ids_from_input_ids_batch(self):
        padding_idx = type(self).padding_idx
        input_ids = torch.tensor(
            [
                [padding_idx, padding_idx, padding_idx, 1, 4, 2, 19, 28],
                [29, 1, 4, 2, 19, 28, 2, 34],
                [padding_idx, 1, 4, 2, 19, 28, 2, 34],
            ],
            dtype=torch.long,
        )

        expected_position_ids = torch.tensor(
            [
                [0, 0, 0, 0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [0, 0, 1, 2, 3, 4, 5, 6],
            ],
            dtype=torch.long,
        )

        actual_position_ids = create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=padding_idx)

        assert torch.equal(actual_position_ids, expected_position_ids)

    def test_last_token_logits_via_padded_position_ids_same_result_as_individual_no_padding(self):
        """Check that when you run batched inference using position_ids determined with create_position_ids_from_input_ids, the output is equivalent to running inference on one example at a time with no padding."""
        model_name = "gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        sentences = ["this has 3 pads", "this sentence has no padding at all", "this sentence has 2 pads"]

        # Padded batch example
        tokens = tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)
        position_ids = create_position_ids_from_input_ids(tokens["input_ids"], padding_idx=model.config.pad_token_id)
        batch_output = model(**tokens, position_ids=position_ids)["logits"][:, -1, :]

        # One-by-one example
        single_output = []
        for sentence in sentences:
            tokens = tokenizer(sentence, padding=False, return_tensors="pt").to(model.device)
            # position_ids = create_position_ids_from_input_ids(tokens["input_ids"], padding_idx=model.config.pad_token_id)
            single_output.append(model(**tokens)["logits"][:, -1, :])
        single_output = torch.cat(single_output, dim=0)

        assert torch.allclose(batch_output, single_output)


class TestScoreModelForNextWordProb(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
        ).to(device)

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name,
            padding_side="left",
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model.config.pad_token_id = cls.model.config.eos_token_id

    def test_score_model_for_next_word_prob_sans_answer_map(self):
        model = type(self).model
        tokenizer = type(self).tokenizer

        prompts = [
            "A is rated ",
            "Item B is rated ",
        ]

        position_ids = torch.tensor(
            [
                [0, 0, 1, 2, 3],
                [0, 1, 2, 3, 4],
            ],
            dtype=torch.long,
            device=model.device,
        )

        actual = score_model_for_next_word_prob(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
        )
        expected = model(
            **tokenizer(prompts, padding=True, return_tensors="pt").to(model.device), position_ids=position_ids
        )["logits"][:, -1, :]
        assert torch.equal(actual, expected)

    def test_score_model_for_next_word_prob_slice(self):
        model = type(self).model
        tokenizer = type(self).tokenizer

        prompts = [
            "A is rated ",
            "Item B is rated ",
        ]

        position_ids = torch.tensor(
            [
                [0, 0, 1, 2, 3],
                [0, 1, 2, 3, 4],
            ],
            dtype=torch.long,
            device=model.device,
        )

        actual = score_model_for_next_word_prob(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            start=0,
            end=1,
        )

        expected = model(
            **tokenizer(prompts, padding=True, return_tensors="pt").to(model.device), position_ids=position_ids
        )["logits"][:, -1, :][:1, :]
        assert torch.allclose(actual, expected)

    def test_score_model_for_next_word_prob_with_answer_map(self):
        model = type(self).model
        tokenizer = type(self).tokenizer

        prompts = [
            "A is rated ",
            "Item B is rated ",
        ]

        position_ids = torch.tensor(
            [
                [0, 0, 1, 2, 3],
                [0, 1, 2, 3, 4],
            ],
            dtype=torch.long,
            device=model.device,
        )

        answer_map_str = {  # noqa: F841
            0: ["zero", "0"],
            1: ["one", "1"],
        }

        answer_map = {
            0: torch.tensor(tokenizer.convert_tokens_to_ids(["zero", "0"]), device=model.device),
            1: torch.tensor(tokenizer.convert_tokens_to_ids(["one", "1"]), device=model.device),
        }

        actual = score_model_for_next_word_prob(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            answer_map=answer_map,
        )
        expected = model(
            **tokenizer(prompts, padding=True, return_tensors="pt").to(model.device), position_ids=position_ids
        )["logits"][:, -1, :]
        expected_aggregated_answer_map = torch.stack(
            [
                expected[:, tokenizer.convert_tokens_to_ids("zero")]
                + expected[:, tokenizer.convert_tokens_to_ids("0")],
                expected[:, tokenizer.convert_tokens_to_ids("one")] + expected[:, tokenizer.convert_tokens_to_ids("1")],
            ],
            dim=1,
        )
        assert torch.equal(actual, expected_aggregated_answer_map)


class TestShardedScoreModel(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
        ).to(device)

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name,
            padding_side="left",
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model.config.pad_token_id = cls.model.config.eos_token_id

    def test_sharded_score_model(self):
        model = type(self).model
        tokenizer = type(self).tokenizer
        prompts = ["A is rated ", "Item B is rated ", "C is also rated ", "The item D is rated as "]

        batch_output = score_model_for_next_word_prob(prompts, model, tokenizer)
        sharded1_output = sharded_score_model(score_model_for_next_word_prob, model, tokenizer, prompts, bs=1)
        sharded2_output = sharded_score_model(score_model_for_next_word_prob, model, tokenizer, prompts, bs=2)
        sharded3_output = sharded_score_model(score_model_for_next_word_prob, model, tokenizer, prompts, bs=3)
        sharded4_output = sharded_score_model(score_model_for_next_word_prob, model, tokenizer, prompts, bs=4)

        assert torch.allclose(batch_output, sharded1_output)
        assert torch.allclose(batch_output, sharded2_output)
        assert torch.allclose(batch_output, sharded3_output)
        assert torch.allclose(batch_output, sharded4_output)


class TestEstimateCMI(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
        ).to(device)

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name,
            padding_side="left",
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model.config.pad_token_id = cls.model.config.eos_token_id

    @patch("measuring.estimate_probs.estimate_prob_x_given_e")
    @patch("measuring.estimate_probs.estimate_prob_y_given_context_and_entity")
    def test_estimate_cmi_when_contexts_have_no_effect_is_0(
        self, mock_estimate_prob_y_given_context_and_entity, mock_estimate_prob_x_given_e
    ):
        model = type(self).model
        tokenizer = type(self).tokenizer

        output_size = 4  # noqa: F841
        num_contexts = 3

        mock_estimate_prob_x_given_e.return_value = torch.ones(num_contexts) / num_contexts  # shape: (3,)
        mock_estimate_prob_y_given_context_and_entity.return_value = torch.tensor(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ]
        )

        contexts = [
            "The movie was great. ",
            "I loved it. ",
            "I hated this. ",
        ]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = "entity1"

        actual = estimate_cmi(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            answer_map=None,
        )
        expected = 0.0

        assert actual == expected

    @patch("measuring.estimate_probs.estimate_prob_x_given_e")
    @patch("measuring.estimate_probs.estimate_prob_y_given_context_and_entity")
    def test_estimate_cmi_when_context_determines_answer_is_entropy(
        self, mock_estimate_prob_y_given_context_and_entity, mock_estimate_prob_x_given_e
    ):
        model = type(self).model
        tokenizer = type(self).tokenizer

        output_size = 4
        num_contexts = 4

        mock_estimate_prob_x_given_e.return_value = torch.ones(num_contexts) / num_contexts
        mock_estimate_prob_y_given_context_and_entity.return_value = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        contexts = ["The movie was great. ", "I loved it. ", "I hated this. ", "I enjoyed it. "]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = "entity1"

        actual = estimate_cmi(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            answer_map=None,
        )
        # I(X; Y | Z) = I(contexts; answers | entity) = H(answers | entity) - H(answers | contexts, entity)
        # The answers distribution is uniform, so the entropy is log(output_size)
        # The answers distribution conditioned on contexts has full prob mass on an output depending on the context, so the entropy is 0 (there's 0 uncertainty in the answer once you know the context).
        # Therefore, the CMI should just be the value of the entropy of the answer distribution
        expected = np.log(output_size) - 0

        assert np.isclose(actual, expected)


class TestEstimateEntityScores(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
        ).to(device)

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name,
            padding_side="left",
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model.config.pad_token_id = cls.model.config.eos_token_id

    @patch("measuring.estimate_probs.estimate_prob_x_given_e")
    @patch("measuring.estimate_probs.score_model_for_next_word_prob")
    def test_estimate_entity_score_when_contexts_have_no_effect_is_0(
        self, mock_score_model_for_next_word_prob, mock_estimate_prob_x_given_e
    ):
        model = type(self).model
        tokenizer = type(self).tokenizer

        output_size = 4  # noqa: F841
        num_contexts = 3

        mock_estimate_prob_x_given_e.return_value = torch.ones(num_contexts) / num_contexts  # shape: (3,)
        mock_score_model_for_next_word_prob.side_effect = [
            torch.tensor([0.25, 0.25, 0.25, 0.25]),
            torch.tensor(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ]
            ),
        ]

        contexts = [
            "The movie was great. ",
            "I loved it. ",
            "I hated this. ",
        ]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = "entity1"

        actual = estimate_entity_score(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            distance_metric=kl_div,
            answer_map=None,
        )
        expected = 0.0

        assert actual == expected

    @patch("measuring.estimate_probs.estimate_prob_x_given_e")
    @patch("measuring.estimate_probs.score_model_for_next_word_prob")
    def test_estimate_entity_score_when_context_determines_answer(
        self, mock_score_model_for_next_word_prob, mock_estimate_prob_x_given_e
    ):
        model = type(self).model
        tokenizer = type(self).tokenizer

        output_size = 4  # noqa: F841
        num_contexts = 3

        mock_estimate_prob_x_given_e.return_value = torch.ones(num_contexts) / num_contexts
        mock_score_model_for_next_word_prob.side_effect = [
            torch.tensor([0.25, 0.25, 0.25, 0.25]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
        ]

        contexts = ["The movie was great. ", "I loved it. ", "I hated this. "]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = "entity1"

        actual = estimate_entity_score(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            distance_metric=kl_div,
            answer_map=None,
        )
        expected = np.dot(
            np.array(
                [
                    np.dot(
                        np.array([1.0, 0.0, 0.0, 0.0]),
                        np.nan_to_num(np.log(np.array([1.0, 0.0, 0.0, 0.0]) / np.array([0.25, 0.25, 0.25, 0.25]))),
                    ),
                    np.dot(
                        np.array([0.0, 1.0, 0.0, 0.0]),
                        np.nan_to_num(np.log(np.array([0.0, 1.0, 0.0, 0.0]) / np.array([0.25, 0.25, 0.25, 0.25]))),
                    ),
                    np.dot(
                        np.array([0.0, 0.0, 1.0, 0.0]),
                        np.nan_to_num(np.log(np.array([0.0, 0.0, 1.0, 0.0]) / np.array([0.25, 0.25, 0.25, 0.25]))),
                    ),
                ]
            ),
            mock_estimate_prob_x_given_e.return_value,
        )  # interestingly, this also happens to equal the MI of ln(4) (entropy of the answer distribution)!

        assert np.isclose(actual, expected)

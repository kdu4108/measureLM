from preprocessing.utils import format_query
import numpy as np
import torch
import unittest as ut
from transformers import AutoModelForCausalLM, AutoTokenizer


class TestFormatQuery(ut.TestCase):
    def test_format_query(self):
        contexts = [
            "The movie was great. ",
            "I loved it. ",
            "I hated this. ",
        ]
        query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
        entity = ("entity1",)

        expected = [
            "The movie was great. On a scale from 1 to 5 stars, the quality of this movie, 'entity1', is rated ",
            "I loved it. On a scale from 1 to 5 stars, the quality of this movie, 'entity1', is rated ",
            "I hated this. On a scale from 1 to 5 stars, the quality of this movie, 'entity1', is rated ",
        ]
        actual = [format_query(entity=entity, context=context, query=query) for context in contexts]
        assert expected == actual

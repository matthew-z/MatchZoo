"""
These tests are simplied because the original verion takes too much time to
run, making CI fails as it reaches the time limit.
"""
from pathlib import Path

import pytest

import matchzoo as mz
from matchzoo import preprocessors


@pytest.fixture(scope='module',
                params=preprocessors.list_available())
def preprocessor_cls(request):
    return request.param


@pytest.fixture(scope='module', params=[
    mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=2)),
    mz.tasks.Classification(num_classes=2),
])
def task(request):
    return request.param


@pytest.fixture(scope='module')
def train_raw(task):
    return mz.datasets.toy.load_data('train', task)[:5]


def test_fit_transform(train_raw, preprocessor_cls):
    if preprocessor_cls != preprocessors.BertPreprocessor:
        p1 = preprocessor_cls(multiprocessing=False)
        p2 = preprocessor_cls(multiprocessing=True)
    else:

        bert_vocab = Path(mz.datasets.__file__).parent.joinpath(
            "bert_resources/uncased_vocab_100.txt")

        p1 = preprocessor_cls(
            bert_vocab_path=bert_vocab,
            multiprocessing=False)
        p2 = preprocessor_cls(
            bert_vocab_path=bert_vocab,
            multiprocessing=True)

    processed = p1.fit_transform(train_raw)
    assert processed

    processed_parallel = p2.fit_transform(train_raw)
    assert processed_parallel

    assert all(processed.left == processed_parallel.left)
    assert all(processed.right == processed_parallel.right)
    assert all(processed.relation == processed_parallel.relation)

from . import units
from .basic_preprocessor import BasicPreprocessor
from .bert_preprocessor import BertPreprocessor
from .cdssm_preprocessor import CDSSMPreprocessor
from .diin_preprocessor import DIINPreprocessor
from .dssm_preprocessor import DSSMPreprocessor
from .naive_preprocessor import NaivePreprocessor


def list_available() -> list:
    from matchzoo.engine.base_preprocessor import BasePreprocessor
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BasePreprocessor)

from typing import Dict, Tuple, List
import pytest

from dataclasses import dataclass, field

from br2_vision.data_structure.io_utils import DataclassYamlSaveLoadMixin


@dataclass
class MockClass(DataclassYamlSaveLoadMixin):
    name: str = "John"
    age: int = 30

    list_type: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    dictionary_type: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {"key1": (0.0, 0.0, 0.0), "key2": (0.0, 0.0, 0.0)})
    dictionary_type2: Dict[str, List[float]] = field(default_factory=lambda: {"key1": [0.0, 0.0, 0.0], "key2": [0.0, 0.0, 0.0]})
    tuple_type: Tuple[float, float, float] = (0.0, 0.0, 0.0)

def test_save_load_yaml(tmp_path):
    file_path = tmp_path / "test.yaml"
    obj = MockClass()
    obj.to_yaml(file_path)
    obj2 = MockClass.from_yaml(file_path)
    assert obj == obj2
    assert obj.name == obj2.name
    assert obj.age == obj2.age
    assert obj is not obj2

    assert obj.list_type == obj2.list_type
    assert obj.dictionary_type == obj2.dictionary_type
    print(obj.dictionary_type)
    print(obj2.dictionary_type)
    assert obj.tuple_type == obj2.tuple_type

def test_no_file_case():
    with pytest.raises(FileNotFoundError):
        MockClass.from_yaml("no_file.yaml")

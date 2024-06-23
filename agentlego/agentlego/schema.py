import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type


@dataclass
class Parameter:
    """Meta information for parameters.

    Args:
        type (type): The type of the value.
        name (str): tool name for agent to identify the tool.
        description (str): Description for the parameter.
        optional (bool): Whether the parameter has a default value.
            Defaults to False.
        default (Any): The default value of the parameter.
    """
    type: Optional[Type] = None
    name: Optional[str] = None
    description: Optional[str] = None
    optional: Optional[bool] = None
    default: Optional[Any] = None
    filetype: Optional[str] = None

    def update(self, other: 'Parameter'):
        other = copy.deepcopy(other)
        for k, v in copy.deepcopy(other.__dict__).items():
            if v is not None:
                self.__dict__[k] = v

    def to_json_dict(self) -> dict:
        '''Return a serializable dict'''
        from .types import CatgoryToIO
        data = self.__dict__.copy()
        if data['type'] is not None:
            data['type'] = {v: k for k, v in CatgoryToIO.items()}[data['type']]
        return data

    @classmethod
    def from_json_dict(self, json_dict: dict) -> 'Parameter':
        '''De-serialize from a dumped dict'''
        from .types import CatgoryToIO
        data = {**json_dict, 'type': CatgoryToIO[json_dict['type']]}
        return Parameter(**data)


@dataclass
class ToolMeta:
    """Meta information for tool.

    Args:
        name (str): tool name for agent to identify the tool.
        description (str): Description for tool.
        inputs (tuple[str | Parameter, ...]): Input categories for tool.
        outputs (tuple[str | Parameter, ...]): Output categories for tool.
    """
    name: Optional[str] = None
    description: Optional[str] = None
    inputs: Optional[Tuple[Parameter, ...]] = None
    outputs: Optional[Tuple[Parameter, ...]] = None

    def to_json_dict(self) -> dict:
        '''Return a serializable dict'''
        data = self.__dict__.copy()
        if data['inputs'] is not None:
            data['inputs'] = tuple(p.to_json_dict() for p in data['inputs'])
        if data['outputs'] is not None:
            data['outputs'] = tuple(p.to_json_dict() for p in data['outputs'])
        return data

    @classmethod
    def from_json_dict(self, json_dict: dict) -> 'ToolMeta':
        '''De-serialize from a dumped dict'''
        data = json_dict.copy()
        if json_dict['inputs'] is not None:
            data['inputs'] = tuple(Parameter.from_json_dict(item) for item in json_dict['inputs'])
        if json_dict['outputs'] is not None:
            data['outputs'] = tuple(Parameter.from_json_dict(item) for item in json_dict['outputs'])
        return ToolMeta(**data)

"""
Data adapters for different motion capture datasets.

This module provides adapters for loading and processing motion data from different sources.
"""

from .base import BaseAdapter
from .amass import AMASSAdapter
from .bandai import BANDAIAdapter
from .mujoco import MuJoCoAdapter

__all__ = ['AMASSAdapter', 'BANDAIAdapter', 'MuJoCoAdapter', 'AdapterRegistry', 'get_adapter_for_character']

class AdapterRegistry:
    CHARACTER_ADAPTER_MAP = {
        BANDAIAdapter : ['character2'],
        AMASSAdapter : ["Karim", "Aude", "Medhi", "Carine", 'Rub'],
        MuJoCoAdapter : ["humanoid"]
    }

    @classmethod
    def get_adapter(cls, character_name: str, device: str):
        for adapter, chars in cls.CHARACTER_ADAPTER_MAP.items():
            if character_name in chars:
                return adapter(device)
        return None

    @classmethod
    def list_characters(cls, adapter: BaseAdapter):
        return cls.CHARACTER_ADAPTER_MAP[type(adapter)]


def get_adapter_for_character(character_name: str, device: str) -> BaseAdapter:
    return AdapterRegistry.get_adapter(character_name, device)


def list_characters(adapter: BaseAdapter) -> str:
    return AdapterRegistry.list_characters(adapter)

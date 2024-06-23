import importlib
from typing import List


def try_import_RMSNorm():
    """
    Overview:
        Try import RMSNorm module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module, or ``None`` when RMSNorm not found
    """
    try:
        from apex.normalization.fused_layer_norm import MixedFusedRMSNorm as RMSNorm

        return RMSNorm
    except ModuleNotFoundError as e:
        print(f"RMSNorm package import error! {e}", flush=True)
        return None


def try_import_ceph():
    """
    Overview:
        Try import boto3 module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module, or ``None`` when boto3 not found
    """
    try:
        import boto3

        return boto3
    except ModuleNotFoundError as e:
        print(f"RMSNorm package import error! {e}", flush=True)
        return None


def try_import_petrel_client():
    """
    Overview:
        Try import petrel_client module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module,
            or ``None`` when petrel_client not found
    """
    try:
        from petrel_client.client import Client

        Client()
        return Client
    except Exception as e:
        print(f"petrel_client.client import error! {e}", flush=True)
        return lambda *args, **kwargs: None


def try_import_botocore():
    """
    Overview:
        Try import petrel_client module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module, or ``None`` when petrel_client not found
    """
    try:
        import botocore

        return botocore
    except ModuleNotFoundError as e:
        print(f"RMSNorm package import error! {e}", flush=True)
        return None


def try_import_oss2():
    """
    Overview:
        Try import petrel_client module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module, or ``None`` when petrel_client not found
    """
    try:
        import oss2

        return oss2
    except ModuleNotFoundError as e:
        print(f"RMSNorm package import error! {e}", flush=True)
        return None


def import_module(modules: List[str]) -> None:
    """
    Overview:
        Import several module as a list
    Arguments:
        - (:obj:`str list`): List of module names
    """
    for name in modules:
        importlib.import_module(name)

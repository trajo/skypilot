"""Durable count of the Nebius VMs each cluster created.

Bounds creations under nebius.max_instance_creations across failover
candidates and managed-job recoveries; see sky.provision.creation_journal.
A cluster may create several VMs at once (head and workers in parallel), so
all of them are reserved together before the first create.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from sky.provision import creation_journal

ROOT = Path.home() / '.sky' / 'nebius-provisioning'
_PROVIDER = 'Nebius'


def read(name: str) -> Optional[Dict[str, Any]]:
    return creation_journal.read(ROOT, name)


def register(cluster: str, name: str) -> None:
    creation_journal.register(ROOT, _PROVIDER, cluster, name)


def reserve(name: str, count: int, max_instance_creations: int) -> List[str]:
    return creation_journal.reserve(ROOT, _PROVIDER, name, count,
                                    max_instance_creations)


def resolve(name: str, outcomes: Dict[str, Optional[str]]) -> None:
    creation_journal.resolve(ROOT, name, outcomes)

"""Durable per-cluster records of instance creations, for any cloud.

Each cloud keeps one JSON file per cluster under its own directory, named by
the SHA-256 of the cluster's name on the cloud. The record is kept outside
cluster metadata, so status refresh and teardown cannot erase it, and it is
never deleted: a managed job keeps its cluster name across recoveries, so the
record counts every instance the job created.

Rules for callers that bound creations:
  - reserve a slot (pending) durably before every create call;
  - count the slot once the create is confirmed, or its outcome is unknown;
  - release the slot only when the provider definitely refused the create;
  - fail closed on a missing record or an unresolved pending slot;
  - raise exceptions.InstanceCreationLimitError at the limit.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import uuid

from sky import exceptions
from sky.utils import common_utils


def path(root: Path, name: str) -> Path:
    return root / (hashlib.sha256(name.encode()).hexdigest() + '.json')


def read(root: Path, name: str) -> Optional[Dict[str, Any]]:
    record = path(root, name)
    return json.loads(record.read_text()) if record.exists() else None


def write(root: Path, value: Dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as out:
        json.dump(value, out, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    os.replace(out.name, path(root, value['name']))
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def register(root: Path, provider: str, cluster: str, name: str,
             **fields: Any) -> None:
    """Create the cluster's record, or check it belongs to this cluster."""
    value = read(root, name)
    if value is None:
        write(
            root,
            dict(cluster=cluster,
                 name=name,
                 user=common_utils.get_user_hash(),
                 ids=[],
                 pending=None,
                 **fields))
    elif value['cluster'] != cluster or value[
            'user'] != common_utils.get_user_hash():
        raise RuntimeError(f'{provider} provisioning identity mismatch')


def check_limit(root: Path,
                provider: str,
                name: str,
                max_instance_creations: Optional[int] = None,
                count: int = 1) -> Optional[Dict[str, Any]]:
    """Raise if the cluster may not create `count` more instances.

    Writes nothing. Returns the record, or None if there is none.
    """
    value = read(root, name)
    if max_instance_creations is not None:
        if (isinstance(max_instance_creations, bool) or
                not isinstance(max_instance_creations, int) or
                max_instance_creations < 1):
            raise ValueError(
                'max_instance_creations must be a positive integer')
        if value is None:
            raise RuntimeError(
                f'{provider} creation limit requires a provisioning journal')
        if len(value['ids']) + count > max_instance_creations:
            raise exceptions.InstanceCreationLimitError(
                f'{provider} instance creation limit reached; '
                'submit a new authorized request')
    return value


def reserve(root: Path, provider: str, name: str, count: int,
            max_instance_creations: int) -> List[str]:
    """Durably reserve `count` creations, all or none, before any create.

    Returns one slot per create, to be passed to resolve(). Raises at the
    limit, and while slots of an earlier reservation are unresolved (their
    creates may have run).
    """
    value = check_limit(root, provider, name, max_instance_creations, count)
    if value is None:
        raise RuntimeError(
            f'{provider} creation limit requires a provisioning journal')
    if value['pending']:
        raise RuntimeError(f'Unresolved {provider} create; reconcile before '
                           f'retry ({path(root, name)})')
    slots = [uuid.uuid4().hex for _ in range(count)]
    value['pending'] = slots
    write(root, value)
    return slots


def resolve(root: Path, name: str, outcomes: Dict[str, Optional[str]]) -> None:
    """Settle reserved slots: count each as its identity, or release None.

    Count a create once it is confirmed or its outcome is unknown (identify it
    by instance ID, or by name if the ID is unknown). Release (None) only a
    create that never ran or that the provider definitely refused.
    """
    value = read(root, name)
    if value is None:
        raise RuntimeError(f'Provisioning journal for {name} is missing')
    pending = list(value['pending'] or [])
    for slot, identity in outcomes.items():
        if slot in pending:
            pending.remove(slot)
            if identity is not None:
                value['ids'].append(str(identity))
    value['pending'] = pending or None
    write(root, value)


def for_cluster(root: Path, cluster: str) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    return [
        value for record in root.glob('*.json')
        if (value := json.loads(record.read_text()))['cluster'] == cluster and
        value['user'] == common_utils.get_user_hash()
    ]

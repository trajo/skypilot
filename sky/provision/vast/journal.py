"""Durable identities for Vast allocations interrupted before registration.

Kept outside cluster metadata so status refresh cannot erase cleanup evidence.
Unknown create outcomes hold cleanup open until an exact label is discovered or
an operator reconciles the request. Empty inventory cannot resolve them.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from sky.utils import common_utils

ROOT = Path.home() / '.sky' / 'vast-provisioning'


def _path(name):
    return ROOT / (hashlib.sha256(name.encode()).hexdigest() + '.json')


def read(name):
    path = _path(name)
    return json.loads(path.read_text()) if path.exists() else None


def write(value):
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode='w', dir=ROOT, delete=False) as out:
        json.dump(value, out, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    os.replace(out.name, _path(value['name']))
    directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def register(cluster, name):
    value = read(name)
    if value is None:
        write(
            dict(cluster=cluster,
                 name=name,
                 user=common_utils.get_user_hash(),
                 ids=[],
                 pending=None,
                 cleaned=False))
    elif value['cluster'] != cluster or value[
            'user'] != common_utils.get_user_hash():
        raise RuntimeError('Vast provisioning identity mismatch')


def creating(label):
    name = label.rsplit('-', 1)[0]
    value = read(name)
    if value is not None:
        if value['pending']:
            raise RuntimeError('Unresolved Vast create; reconcile before retry')
        value.update(pending=label, cleaned=False)
        write(value)


def created(label, instance_id):
    value = read(label.rsplit('-', 1)[0])
    if value is not None:
        value['ids'].append(str(instance_id))
        value['pending'] = None
        write(value)


def for_cluster(cluster):
    if not ROOT.exists():
        return []
    return [
        value for path in ROOT.glob('*.json')
        if (value := json.loads(path.read_text()))['cluster'] == cluster and
        value['user'] == common_utils.get_user_hash()
    ]


def cleanup_cluster(cluster):
    # Import locally to avoid the instance/utils/journal cycle.
    # pylint: disable=import-outside-toplevel
    from sky.provision.vast import instance
    for value in for_cluster(cluster):
        instance.terminate_instances(value['name'])

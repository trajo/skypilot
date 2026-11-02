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
import time

from sky import exceptions
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


def creating(label, max_instance_creations=None):
    name = label.rsplit('-', 1)[0]
    value = read(name)
    if max_instance_creations is not None:
        if (isinstance(max_instance_creations, bool) or
                not isinstance(max_instance_creations, int) or
                max_instance_creations < 1):
            raise ValueError(
                'max_instance_creations must be a positive integer')
        if value is None:
            raise RuntimeError(
                'Vast creation limit requires a provisioning journal')
        if len(value['ids']) >= max_instance_creations:
            raise exceptions.ProvisionUnsupportedError(
                'Vast instance creation limit reached; '
                'submit a new authorized request')
    if value is not None:
        if value['pending']:
            raise RuntimeError('Unresolved Vast create; reconcile before retry')
        value.update(pending=label, cleaned=False)
        write(value)


def created(label, instance_id, offer=None):
    value = read(label.rsplit('-', 1)[0])
    if value is not None:
        value['ids'].append(str(instance_id))
        if offer is not None:
            # Which host each instance came from, for excluding bad hosts.
            value.setdefault('offers', {})[str(instance_id)] = offer
        value['pending'] = None
        write(value)


def rejected(label, status):
    """The provider refused a single-attempt create: nothing was created."""
    value = read(label.rsplit('-', 1)[0])
    if value is not None and value['pending'] == label:
        value.setdefault('rejections', []).append(int(status))
        value['pending'] = None
        write(value)


def refused(label, offer):
    """Remember an offer the provider would not rent to this cluster."""
    value = read(label.rsplit('-', 1)[0])
    if value is not None:
        value.setdefault('refused_offers', {})[str(offer)] = time.time()
        write(value)


def recently_refused(label, seconds):
    """Offers refused to this cluster within the last `seconds`."""
    value = read(label.rsplit('-', 1)[0]) or {}
    cutoff = time.time() - seconds
    return {
        offer for offer, at in value.get('refused_offers', {}).items()
        if at >= cutoff
    }


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

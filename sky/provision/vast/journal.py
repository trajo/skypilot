"""Durable identities for Vast allocations interrupted before registration.

Kept outside cluster metadata so status refresh cannot erase cleanup evidence.
Unknown create outcomes hold cleanup open until an exact label is discovered or
an operator reconciles the request. Empty inventory cannot resolve them.

Storage, identity and the creation limit are shared with other clouds (see
sky.provision.creation_journal). Vast-specific here: one pending create at a
time, keyed by instance label, rejected creates, and the hosts and offers each
instance came from or was refused on.
"""
from pathlib import Path
import time

from sky.provision import creation_journal
# Tests patch common_utils through this module.
from sky.utils import common_utils  # pylint: disable=unused-import

ROOT = Path.home() / '.sky' / 'vast-provisioning'
_PROVIDER = 'Vast'


def read(name):
    return creation_journal.read(ROOT, name)


def write(value):
    creation_journal.write(ROOT, value)


def register(cluster, name):
    creation_journal.register(ROOT, _PROVIDER, cluster, name, cleaned=False)


def check_limit(label, max_instance_creations=None):
    """Raise if the cluster may not create another instance. Writes nothing.

    Returns the journal record, or None if there is none.
    """
    return creation_journal.check_limit(ROOT, _PROVIDER,
                                        label.rsplit('-', 1)[0],
                                        max_instance_creations)


def creating(label, max_instance_creations=None):
    value = check_limit(label, max_instance_creations)
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
    return creation_journal.for_cluster(ROOT, cluster)


def cleanup_cluster(cluster):
    # Import locally to avoid the instance/utils/journal cycle.
    # pylint: disable=import-outside-toplevel
    from sky.provision.vast import instance
    for value in for_cluster(cluster):
        instance.terminate_instances(value['name'])

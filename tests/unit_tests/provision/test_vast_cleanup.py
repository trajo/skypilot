"""No successful cleanup without exact independent resource absence."""
from unittest import mock

import pytest

from sky.provision.vast import instance
from sky.provision.vast import journal
from sky.provision.vast import utils


@pytest.fixture(autouse=True)
def isolated_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, 'ROOT', tmp_path)
    monkeypatch.setattr(journal.common_utils, 'get_user_hash', lambda: 'owner')
    monkeypatch.setattr(instance.time, 'sleep', lambda _: None)
    journal.register('display', 'cloud')


def vm(label='cloud-head'):
    return {'id': '123', 'name': label, 'status': 'RUNNING'}


def test_partially_registered_and_repeated_cleanup(monkeypatch):
    journal.creating('cloud-head')
    journal.created('cloud-head', '123')
    inventory = mock.Mock(side_effect=[{
        '123': vm(),
        '456': vm('other-head')
    }, {
        '456': vm('other-head')
    }, {}, {}])
    monkeypatch.setattr(utils, 'list_instances', inventory)
    remove = mock.Mock()
    monkeypatch.setattr(utils, 'remove', remove)
    journal.cleanup_cluster('display')
    journal.cleanup_cluster('display')
    remove.assert_called_once_with('123')
    assert journal.read('cloud')['cleaned']


def test_unknown_create_empty_inventory_is_not_cleanup(monkeypatch):
    journal.creating('cloud-head')
    monkeypatch.setattr(utils, 'list_instances', lambda: {})
    with pytest.raises(RuntimeError, match='Unresolved'):
        journal.cleanup_cluster('display')
    assert not journal.read('cloud')['cleaned']


def test_lost_create_response_reconciles_exact_label(monkeypatch):
    journal.creating('cloud-head')
    monkeypatch.setattr(utils, 'list_instances',
                        mock.Mock(side_effect=[{
                            '123': vm()
                        }, {}]))
    remove = mock.Mock()
    monkeypatch.setattr(utils, 'remove', remove)
    journal.cleanup_cluster('display')
    assert journal.read('cloud')['ids'] == ['123']
    assert journal.read('cloud')['cleaned']


def test_delete_acknowledgement_does_not_mean_absent(monkeypatch):
    monkeypatch.setattr(utils, 'list_instances', lambda: {'123': vm()})
    monkeypatch.setattr(utils, 'remove', lambda _: {'success': True})
    with pytest.raises(RuntimeError, match='remain'):
        journal.cleanup_cluster('display')
    assert not journal.read('cloud')['cleaned']


@pytest.mark.parametrize(
    'response', [None, {}, {
        'instances': None
    }, 'error', [None], [{
        'id': 1
    }]])
def test_ambiguous_inventory_fails_closed(monkeypatch, response):
    sdk = mock.Mock()
    sdk.show_instances.return_value = response
    monkeypatch.setattr(utils.vast, 'vast', lambda: sdk)
    with pytest.raises(RuntimeError):
        utils.list_instances()


def test_unknown_status_is_live(monkeypatch):
    monkeypatch.setattr(utils, 'list_instances',
                        lambda: {'123': dict(vm(), status='UNKNOWN')})
    assert '123' in instance.query_instances('display', 'cloud', {})


def test_cleanup_after_missing_cluster_row(monkeypatch):
    from sky import exceptions
    from sky.jobs import utils as jobs
    monkeypatch.setattr(jobs.global_user_state, 'get_cluster_from_name',
                        lambda _: None)
    monkeypatch.setattr(
        'sky.core.down',
        mock.Mock(side_effect=exceptions.ClusterDoesNotExist('display')))
    monkeypatch.setattr(utils, 'list_instances',
                        mock.Mock(side_effect=[{
                            '123': vm()
                        }, {}]))
    remove = mock.Mock()
    monkeypatch.setattr(utils, 'remove', remove)
    jobs.terminate_cluster('display')
    remove.assert_called_once_with('123')


@pytest.mark.parametrize('direct', [True, False])
def test_ssh_endpoint_is_a_consistent_pair(monkeypatch, direct):
    info = dict(vm(),
                public_ipaddr='1.2.3.4',
                ssh_host='ssh.vast.ai',
                ssh_port=1234,
                local_ipaddrs='10.0.0.1',
                ports={'22/tcp': [{
                    'HostPort': '5678'
                }]} if direct else {})
    monkeypatch.setattr(utils, 'list_instances', lambda: {'123': info})
    result = instance.get_cluster_info('US', 'cloud').instances['123'][0]
    assert (result.external_ip, result.ssh_port) == (('1.2.3.4',
                                                      5678) if direct else
                                                     ('ssh.vast.ai', 1234))


def test_loading_partial_instance_is_not_created_again(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(
        utils, 'list_instances',
        mock.Mock(side_effect=[{
            '123': dict(vm(), status='LOADING')
        }, {
            '123': vm()
        }]))
    launch = mock.Mock()
    monkeypatch.setattr(utils, 'launch', launch)
    config = SimpleNamespace(provider_config={},
                             authentication_config={},
                             docker_config={},
                             node_config={'ImageId': 'image'},
                             count=1,
                             resume_stopped_nodes=False)
    record = instance.run_instances('US', 'display', 'cloud', config)
    assert record.head_instance_id == '123'
    launch.assert_not_called()

"""Interruptible Vast instances: refused bids, outbids and bounded waits.

Observed on Vast (2026-11-02): a bid that only equals an offer's listed
min_bid can be refused with 410 no_such_ask, the same answer as for a vanished
offer, while a higher bid on the same offer succeeds. An outbid instance gets
intended_status 'stopped', also while its image is loading, and stays stopped
until its bid clears the machine's listing and it is started again.
"""
import types
from unittest import mock

import pytest
import requests

from sky.provision.vast import instance
from sky.provision.vast import journal
from sky.provision.vast import utils
from sky.utils import status_lib


@pytest.fixture(autouse=True)
def isolated_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, 'ROOT', tmp_path)
    monkeypatch.setattr(journal.common_utils, 'get_user_hash', lambda: 'owner')
    journal.register('display', 'cloud')


class Clock:
    """Fake time for instance.py: sleeping advances it."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture(name='clock')
def fake_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(
        instance, 'time',
        types.SimpleNamespace(time=clock.time, sleep=clock.sleep))
    return clock


@pytest.fixture(name='offer')
def sample_offer():
    return dict(id=123,
                machine_id=36512,
                gpu_name='RTX 3090',
                num_gpus=1,
                cpu_ram=65536,
                cpu_cores=32,
                cpu_cores_effective=32,
                disk_space=200,
                geolocation='Bucharest, RO',
                verified=True,
                rentable=True,
                rented=False,
                external=False,
                hosting_type=1,
                dph_total=0.25,
                min_bid=0.15,
                storage_cost=0.10)


@pytest.fixture(name='sdk')
def mock_sdk(monkeypatch, offer):
    client = mock.Mock()
    client.search_offers.return_value = [offer]
    client.create_instance.return_value = {'new_contract': 456}
    client.client.api_key = 'test-key'
    monkeypatch.setattr(utils.vast, 'vast', lambda: client)
    monkeypatch.setattr(utils.vast, 'create_sdk', lambda: client)
    monkeypatch.setattr(utils.catalog, 'get_hourly_cost', lambda *a, **kw: 0.25)
    return client


def launch(**kwargs):
    params = dict(name='cloud-head',
                  instance_type='1x-RTX_3090-32-65536',
                  region='RO',
                  disk_size=100,
                  image_name='test-image',
                  ports=None,
                  preemptible=True,
                  secure_only=False)
    params.update(kwargs)
    return utils.launch(**params)


def gone():
    response = requests.Response()
    response.status_code = 410
    return requests.HTTPError('410 Gone', response=response)


@pytest.mark.parametrize('listed,expected', [(0.043, 0.0452), (0.01, 0.012),
                                             (0.28, 0.294)])
def test_bid_clears_the_listing_by_a_margin(listed, expected):
    # pylint: disable=protected-access
    assert utils._bid_over(listed) == pytest.approx(expected)


def test_refused_min_bid_is_retried_once_over_the_listing(sdk, offer):
    sdk.search_offers.side_effect = [[offer], [dict(offer, min_bid=0.16)]]
    sdk.create_instance.side_effect = [gone(), {'new_contract': 456}]
    assert launch() == '456'
    prices = [c.kwargs['price'] for c in sdk.create_instance.call_args_list]
    assert prices == [0.15, pytest.approx(0.168)]
    listing_query = sdk.search_offers.call_args_list[1].kwargs
    assert listing_query['query'] == ('machine_id=36512 rented=any '
                                      'rentable=any')
    assert listing_query['type'] == 'bid'
    record = journal.read('cloud')
    assert record['rejections'] == [410]
    assert record['ids'] == ['456']
    assert record['pending'] is None
    # The journal keeps what the instance actually costs per hour.
    assert record['offers']['456']['hourly'] == pytest.approx(0.168 +
                                                              0.1 * 100 / 720)


@pytest.mark.parametrize('listing', [[], [{'machine_id': 1, 'min_bid': .1}]])
def test_refusal_without_listing_is_not_retried(sdk, offer, listing):
    sdk.search_offers.side_effect = [[offer], listing]
    sdk.create_instance.side_effect = gone()
    with pytest.raises(requests.HTTPError):
        launch()
    sdk.create_instance.assert_called_once()
    assert journal.read('cloud')['pending'] is None


def test_refused_bid_is_not_raised_over_the_ceiling(sdk, offer):
    # 0.15 + 100 GB disk (0.0139/h) fits 0.17; once the listing moved to
    # 0.155, the retry bid 0.1628 with disk does not.
    sdk.search_offers.side_effect = [[offer], [dict(offer, min_bid=0.155)]]
    sdk.create_instance.side_effect = gone()
    with pytest.raises(requests.HTTPError):
        launch(max_hourly_cost=0.17)
    sdk.create_instance.assert_called_once()


@pytest.mark.parametrize('kwargs', [{
    'preemptible': False
}, {
    'create_instance_kwargs': {
        'price': 0.2
    }
}])
def test_only_listed_minimum_bids_are_retried(sdk, offer, kwargs):
    sdk.search_offers.side_effect = [[offer], [dict(offer, min_bid=0.16)]]
    sdk.create_instance.side_effect = gone()
    with pytest.raises(requests.HTTPError):
        launch(**kwargs)
    sdk.create_instance.assert_called_once()


def outbid(**changes):
    return dict(
        dict(id='789',
             machine_id=36512,
             is_bid=True,
             dph_base=0.048,
             dph_total=0.05,
             disk_space=10), **changes)


def test_rebid_clears_the_competing_bid(sdk, offer):
    sdk.search_offers.return_value = [dict(offer, min_bid=0.05)]
    utils.rebid_to_win(outbid(), limit=0.1)
    sdk.change_bid.assert_called_once_with(id=789, price=pytest.approx(0.0525))


@pytest.mark.parametrize('listing', [[{
    'machine_id': 36512,
    'min_bid': 0.048
}], []])
def test_no_rebid_when_already_highest_or_unlisted(sdk, listing):
    sdk.search_offers.return_value = listing
    utils.rebid_to_win(outbid(), limit=0.1)
    sdk.change_bid.assert_not_called()


def test_rebid_never_exceeds_the_limit(sdk, offer):
    sdk.search_offers.return_value = [dict(offer, min_bid=0.07)]
    # 0.0735 bid + 0.002 disk > 0.075.
    with pytest.raises(RuntimeError, match='over the \\$0.0750/hour limit'):
        utils.rebid_to_win(outbid(), limit=0.075)
    sdk.change_bid.assert_not_called()


def vm(**changes):
    return dict(
        dict(id='123',
             name='cloud-head',
             status='RUNNING',
             intended_status='running',
             is_bid=True,
             ssh_port=22), **changes)


def config(**provider):
    return types.SimpleNamespace(provider_config=provider,
                                 authentication_config={},
                                 docker_config={},
                                 node_config={
                                     'ImageId': 'image',
                                     'InstanceType': '1x-RTX_3090-32-65536',
                                     'DiskSize': 10,
                                     'Preemptible': True
                                 },
                                 count=1,
                                 resume_stopped_nodes=True,
                                 ports_to_open_on_launch=None)


@pytest.fixture(name='provider')
def provider_calls(monkeypatch):
    calls = types.SimpleNamespace(rebid=mock.Mock(),
                                  start=mock.Mock(),
                                  launch=mock.Mock(return_value='123'),
                                  limit=mock.Mock(return_value=(0.4, 0.3)))
    monkeypatch.setattr(utils, 'rebid_to_win', calls.rebid)
    monkeypatch.setattr(utils, 'start', calls.start)
    monkeypatch.setattr(utils, 'launch', calls.launch)
    monkeypatch.setattr(utils, 'hourly_limit', calls.limit)
    return calls


def inventory(monkeypatch, *states):
    """list_instances answers each poll with the next state, then the last."""
    states = list(states)

    def answer():
        state = states.pop(0) if len(states) > 1 else states[0]
        return {k: dict(v) for k, v in state.items()}

    monkeypatch.setattr(utils, 'list_instances', answer)


def test_outbid_while_loading_is_won_back_not_awaited(monkeypatch, clock,
                                                      provider):
    # CREATED (image loaded) but intended 'stopped': without a rebid and a
    # start Vast never runs it, so waiting for it as pending would hang.
    loaded_outbid = {'123': vm(status='CREATED', intended_status='stopped')}
    inventory(monkeypatch, loaded_outbid, loaded_outbid, {'123': vm()})
    record = instance.run_instances('RO', 'display', 'cloud',
                                    config(max_hourly_cost=0.4))
    assert record.head_instance_id == '123'
    provider.rebid.assert_called_once()
    assert provider.rebid.call_args.args[1] == 0.4
    provider.start.assert_called_once_with('123')
    provider.launch.assert_not_called()
    assert clock.now - 1000 < 60


def test_user_bid_is_kept_on_resume(monkeypatch, clock, provider):
    del clock
    inventory(monkeypatch,
              {'123': vm(status='EXITED', intended_status='stopped')},
              {'123': vm()})
    instance.run_instances('RO', 'display', 'cloud',
                           config(create_instance_kwargs={'bid_price': 0.2}))
    provider.rebid.assert_not_called()
    provider.start.assert_called_once_with('123')


def test_outbid_again_during_wait_is_won_back_then_failed_over(
        monkeypatch, clock, provider):
    del clock
    # Created fresh, then outbid while loading every time it is restarted.
    inventory(monkeypatch, {}, {'123': vm(status='LOADING')},
              {'123': vm(status='LOADING', intended_status='stopped')})
    with pytest.raises(RuntimeError, match='stopped again after 2 rebids'):
        instance.run_instances('RO', 'display', 'cloud', config())
    provider.launch.assert_called_once()
    assert provider.rebid.call_count == instance.MAX_REBIDS
    assert provider.start.call_count == instance.MAX_REBIDS


def test_started_instance_that_stays_exited_counts_as_outbid(
        monkeypatch, clock, provider):
    del clock
    # Vast could not run it with its bid (another renter holds the machine).
    inventory(monkeypatch,
              {'123': vm(status='EXITED', intended_status='stopped')},
              {'123': vm(status='EXITED')})
    with pytest.raises(RuntimeError, match='stopped again'):
        instance.run_instances('RO', 'display', 'cloud', config())
    assert provider.start.call_count == 1 + instance.MAX_REBIDS


def test_rebid_over_limit_fails_over(monkeypatch, clock, provider):
    del clock
    provider.rebid.side_effect = RuntimeError('over the limit')
    inventory(monkeypatch,
              {'123': vm(status='EXITED', intended_status='stopped')})
    with pytest.raises(RuntimeError, match='over the limit'):
        instance.run_instances('RO', 'display', 'cloud', config())
    provider.start.assert_not_called()


def test_resumed_instance_wait_is_bounded(monkeypatch, clock, provider):
    # Started, then never ready (here: reloading its image without end).
    inventory(monkeypatch,
              {'123': vm(status='EXITED', intended_status='stopped')},
              {'123': vm(status='LOADING')})
    with pytest.raises(RuntimeError, match='not ready'):
        instance.run_instances('RO', 'display', 'cloud', config())
    provider.start.assert_called_once_with('123')
    assert instance.RESUME_TIMEOUT <= clock.now - 1000 < (
        instance.RESUME_TIMEOUT + 2 * instance.POLL_INTERVAL)


def test_loading_wait_is_bounded(monkeypatch, clock, provider):
    del provider
    inventory(monkeypatch, {'123': vm(status='LOADING')})
    with pytest.raises(RuntimeError, match='still pending'):
        instance.run_instances('RO', 'display', 'cloud', config())
    assert clock.now - 1000 <= (instance.LOADING_TIMEOUT +
                                2 * instance.POLL_INTERVAL)


@pytest.mark.parametrize(
    'state,expected',
    [
        (vm(status='CREATED',
            intended_status='stopped'), status_lib.ClusterStatus.STOPPED),
        (vm(status='LOADING',
            intended_status='stopped'), status_lib.ClusterStatus.STOPPED),
        (vm(status='LOADING'), status_lib.ClusterStatus.INIT),
        (vm(status='EXITED'), status_lib.ClusterStatus.STOPPED),
        (vm(), status_lib.ClusterStatus.UP),
        # Outbid while running: SSH fails within seconds, but the instance is
        # stopping, not broken; recovery can restart it in place.
        (vm(intended_status='stopped'), status_lib.ClusterStatus.STOPPED),
    ])
def test_status_of_outbid_instances(monkeypatch, state, expected):
    monkeypatch.setattr(utils, 'list_instances', lambda: {'123': state})
    statuses = instance.query_instances('display', 'cloud', {})
    assert statuses == {'123': (expected, None)}

"""Vast selection must fail closed before any instance creation."""
import copy
import importlib
from unittest import mock

import jsonschema
import pytest

from sky.provision.vast import utils
from sky.skylet import constants
from sky.utils import schemas


@pytest.fixture(name='offer')
def sample_offer():
    return dict(id=123,
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
    client.show_instance.return_value = {'id': '456'}
    client.client.api_key = 'test-key'
    monkeypatch.setattr(utils.vast, 'vast', lambda: client)
    monkeypatch.setattr(utils.catalog, 'get_hourly_cost', lambda *a, **kw: 0.25)
    return client


def launch(**kwargs):
    params = dict(name='test',
                  instance_type='1x-RTX_3090-32-65536',
                  region='RO',
                  disk_size=100,
                  image_name='test-image',
                  ports=None,
                  preemptible=False,
                  secure_only=False)
    params.update(kwargs)
    return utils.launch(**params)


@pytest.mark.parametrize('region,gpu_name', [('RO', 'RTX 3090'),
                                             ('EU', 'RTX PRO 6000 Max-Q')])
def test_query_survives_sdk_processing(sdk, offer, region, gpu_name):
    offer['gpu_name'] = gpu_name
    launch(region=region,
           instance_type=('1x-' + gpu_name.replace(' ', '_') + '-32-65536'))
    query = sdk.search_offers.call_args.kwargs['query']
    # Run through the real installed SDK parser, with no network access.
    try:
        sdk_utils = importlib.import_module('vastai.utils')
    except ImportError:
        legacy_sdk = pytest.importorskip('vastai.vastai_sdk')
        _, args = legacy_sdk.queryParser({'query': query}, None)
        parser = importlib.import_module('vastai.vast')
        processed = args['query']
    else:
        _, _, processed = sdk_utils.preprocess_search_query(query)
        parser = importlib.import_module('vastai.api.query')
        # The reported query loses every constraint in the current SDK.
        original = ('chunked=true georegion=true geolocation="RO" '
                    'disk_space>=100 num_gpus=1 gpu_name="RTX 3090" '
                    'cpu_ram>="64.0"')
        assert sdk_utils.preprocess_search_query(original) == (True, True, '')
    parsed = parser.parse_query(processed, {}, parser.offers_fields,
                                parser.offers_alias, parser.offers_mult)
    assert parsed['gpu_name'] == {'eq': gpu_name}
    assert parsed['num_gpus'] == {'eq': '1'} or parsed['num_gpus'] == {'eq': 1}
    assert parsed['cpu_ram'] == {'gte': 65536.0}
    assert parsed['cpu_cores_effective'] in ({'gte': 32.0}, {'gte': '32.0'})
    assert 'RO' in parsed['geolocation']['in']
    if region == 'RO':
        assert parsed['geolocation'] == {'in': ['RO']}
    assert sdk.search_offers.call_args.kwargs['storage'] == 100
    assert 'chunked' not in query
    assert 'georegion' not in query


@pytest.mark.parametrize('changes', [
    {
        'gpu_name': 'RTX PRO 6000 Max-Q'
    },
    {
        'num_gpus': 2
    },
    {
        'num_gpus': 0
    },
    {
        'cpu_ram': 65535
    },
    {
        'cpu_ram': float('nan')
    },
    {
        'cpu_cores_effective': 16
    },
    {
        'disk_space': 99
    },
    {
        'geolocation': 'US'
    },
    {
        'verified': False
    },
    {
        'rentable': False
    },
    {
        'rented': True
    },
    {
        'external': True
    },
    {
        'gpu_name': None
    },
])
def test_mismatched_offer_never_created(sdk, offer, changes):
    offer.update(changes)
    with pytest.raises(RuntimeError, match='No matching Vast offer'):
        launch()
    sdk.create_instance.assert_not_called()


@pytest.mark.parametrize('field', [
    'gpu_name', 'num_gpus', 'cpu_ram', 'cpu_cores_effective', 'geolocation',
    'disk_space', 'dph_total', 'id'
])
def test_missing_required_field(sdk, offer, field):
    del offer[field]
    with pytest.raises(RuntimeError, match='No matching Vast offer'):
        launch()
    sdk.create_instance.assert_not_called()


@pytest.mark.parametrize('offers', [[], 1, None, [{}]])
def test_no_matching_offer(sdk, offers):
    sdk.search_offers.return_value = offers
    with pytest.raises(RuntimeError, match='No matching Vast offer'):
        launch()
    sdk.create_instance.assert_not_called()


def test_selects_cheapest_matching_offer(sdk, offer):
    wrong = dict(offer, id=1, gpu_name='RTX PRO 6000 Max-Q', dph_total=0.10)
    expensive = dict(offer, id=2, dph_total=0.40)
    sdk.search_offers.return_value = [wrong, expensive, offer]
    assert launch() == '456'
    assert sdk.create_instance.call_args.kwargs['id'] == 123


def test_price_guard_before_creation(sdk, offer):
    offer['dph_total'] = 1.07
    with pytest.raises(RuntimeError, match='exceeds.*limit'):
        launch()
    sdk.create_instance.assert_not_called()
    launch(max_price_multiplier=5)
    sdk.create_instance.assert_called_once()


@pytest.mark.parametrize('price', [None, float('nan'), float('inf'), -1])
def test_invalid_price(sdk, offer, price):
    offer['dph_total'] = price
    with pytest.raises(RuntimeError, match='No matching Vast offer'):
        launch()
    sdk.create_instance.assert_not_called()


def test_datacenter_filter(sdk, offer):
    offer['hosting_type'] = 0
    with pytest.raises(RuntimeError, match='No matching Vast offer'):
        launch(secure_only=True)
    sdk.create_instance.assert_not_called()
    launch(secure_only=False)


def test_continent_and_hyphenated_gpu(sdk, offer):
    offer['gpu_name'] = 'RTX PRO 6000 Max-Q'
    launch(region='EU', instance_type='1x-RTX_PRO_6000_Max-Q-32-65536')
    query = sdk.search_offers.call_args.kwargs['query']
    assert '"RO"' in query
    assert '"RTX PRO 6000 Max-Q"' in query


def test_disk_override_and_spot_bid(sdk):
    launch(preemptible=True,
           create_instance_kwargs={
               'disk': 150,
               'bid_price': 0.20
           })
    assert sdk.search_offers.call_args.kwargs['type'] == 'bid'
    assert sdk.search_offers.call_args.kwargs['storage'] == 150
    assert sdk.create_instance.call_args.kwargs['price'] == 0.20
    sdk.create_instance.reset_mock()
    with pytest.raises(RuntimeError, match='exceeds.*limit'):
        launch(preemptible=True, create_instance_kwargs={'price': 1.07})
    sdk.create_instance.assert_not_called()


def test_template_and_custom_options(sdk):
    launch(
        create_instance_kwargs={
            'template_hash_id': 'template',
            'label': 'custom',
            'env': {
                'X': 'Y'
            },
            'onstart_cmd': 'echo hello',
            'direct': True
        })
    params = sdk.create_instance.call_args.kwargs
    assert params['template_hash'] == 'template'
    assert 'template_hash_id' not in params
    assert params['label'] == 'custom'
    assert params['env'] == {'X': 'Y', '__SOURCE': 'skypilot'}
    assert params['direct'] is True
    assert params['onstart_cmd'].endswith('echo hello')


def test_sdk_does_not_mutate_validation_constraints(sdk, offer):
    original = copy.deepcopy(offer)
    launch()
    assert offer == original
    sdk.create_instance.assert_called_once()


def test_live_price_logged_before_creation(sdk, monkeypatch):
    log = mock.Mock()
    monkeypatch.setattr(utils.logger, 'info', log)

    def create(**_kwargs):
        log.assert_called_once()
        message = log.call_args.args[0]
        assert '$0.2500/hour' in message
        assert '123' in message
        assert 'test-key' not in message
        return {'new_contract': 456}

    sdk.create_instance.side_effect = create
    launch()


def test_disk_allocation_matches_validated_price(sdk):
    launch(create_instance_kwargs={'disk': 10, 'template_hash': 'template'})
    assert sdk.create_instance.call_args.kwargs['disk'] == 100
    assert sdk.search_offers.call_args.kwargs['storage'] == 100


@pytest.mark.parametrize('multiplier', [0, -1, float('nan'), float('inf')])
def test_invalid_price_multiplier(sdk, multiplier):
    with pytest.raises(ValueError, match='max_price_multiplier'):
        launch(max_price_multiplier=multiplier)
    sdk.search_offers.assert_not_called()
    sdk.create_instance.assert_not_called()


def test_historical_zero_spot_estimate(sdk, monkeypatch):
    monkeypatch.setattr(
        utils.catalog, 'get_hourly_cost',
        lambda instance_type, spot, *a, **kw: 0 if spot else 0.25)
    launch(preemptible=True)
    assert sdk.create_instance.call_args.kwargs['price'] == 0.15


@pytest.mark.parametrize('multiplier', [1, 2.0, 5.0])
def test_price_config_schema_and_task_override(multiplier):
    jsonschema.validate({'vast': {
        'max_price_multiplier': multiplier
    }}, schemas.get_config_schema())
    assert (
        'vast',
        'max_price_multiplier') in constants.OVERRIDEABLE_CONFIG_KEYS_IN_TASK

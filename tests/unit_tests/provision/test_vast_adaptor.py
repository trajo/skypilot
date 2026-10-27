"""Exercise the supported SDK and complete, fail-closed v1 inventory."""
from unittest import mock

import pytest

from sky.adaptors import vast


def test_supported_sdk_import(monkeypatch):
    import vastai
    monkeypatch.setattr(vast, '_vast_sdk', None)
    factory = mock.Mock()
    monkeypatch.setattr(vastai, 'VastAI', factory)
    assert vast.vast()._sdk is factory.return_value
    factory.assert_called_once_with()


def test_all_inventory_pages():
    sdk = mock.Mock()
    sdk.show_instances_v1.side_effect = [
        {
            'instances': [{
                'id': 1
            }],
            'next_token': 'page2'
        },
        {
            'instances': [{
                'id': 2
            }],
            'next_token': None
        },
    ]
    assert vast._VastClient(sdk).show_instances() == [{'id': 1}, {'id': 2}]
    assert sdk.show_instances_v1.call_args.args[0]['after_token'] == 'page2'
    sdk.show_instances.assert_not_called()


@pytest.mark.parametrize('page', [
    None, {}, {
        'instances': None
    }, {
        'instances': [],
        'next_token': ''
    }, {
        'instances': [],
        'next_token': 1
    }
])
def test_inventory_page_fails_closed(page):
    sdk = mock.Mock()
    sdk.show_instances_v1.return_value = page
    with pytest.raises(RuntimeError, match='Ambiguous'):
        vast._VastClient(sdk).show_instances()


def test_repeated_pagination_token_fails_closed():
    sdk = mock.Mock()
    sdk.show_instances_v1.return_value = {'instances': [], 'next_token': 'same'}
    with pytest.raises(RuntimeError, match='pagination'):
        vast._VastClient(sdk).show_instances()
    assert sdk.show_instances_v1.call_count == 2


def test_inventory_error_propagates():
    sdk = mock.Mock()
    sdk.show_instances_v1.side_effect = RuntimeError('HTTP 410')
    with pytest.raises(RuntimeError, match='410'):
        vast._VastClient(sdk).show_instances()


def test_empty_inventory_is_valid():
    sdk = mock.Mock()
    sdk.show_instances_v1.return_value = {'instances': [], 'next_token': None}
    assert vast._VastClient(sdk).show_instances() == []


def test_real_sdk_uses_v1_inventory():
    import vastai
    sdk = vastai.VastAI(api_key='synthetic-test-only')
    response = mock.Mock()
    response.json.return_value = {'instances': [], 'next_token': None}
    with mock.patch.object(sdk.client, 'get', return_value=response) as get:
        assert vast._VastClient(sdk).show_instances() == []
    assert get.call_args.args == ('/api/v1/instances/',)
    response.raise_for_status.assert_called_once_with()

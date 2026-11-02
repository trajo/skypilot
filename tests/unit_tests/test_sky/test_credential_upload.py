"""Local credential files must be excludable from every cluster."""
import contextlib
import pathlib
from unittest import mock

import jsonschema
import pytest

from sky import check as sky_check
from sky import clouds
from sky import resources as resources_lib
from sky import skypilot_config
from sky.backends import backend_utils
from sky.utils import common_utils
from sky.utils import registry
from sky.utils import schemas


def _no_upload_clouds(region):
    """The clouds backend_utils.write_cluster_config excludes from upload."""
    return {
        cloud for name, cloud in registry.CLOUD_REGISTRY.items()
        if skypilot_config.get_effective_workspace_region_config(
            cloud=name.lower(),
            region=region,
            keys=('remote_identity',),
            default_value=None) == schemas.RemoteIdentityOptions.NO_UPLOAD.value
    }


def test_runpod_remote_identity_config_schema():
    jsonschema.validate({'runpod': {
        'remote_identity': 'NO_UPLOAD'
    }}, schemas.get_config_schema())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'runpod': {
            'unknown': 1
        }}, schemas.get_config_schema())


def test_runpod_no_upload_excludes_its_credentials(monkeypatch, tmp_path):
    key_file = tmp_path / 'config.toml'
    key_file.write_text('[default]\napi_key = "k"\n')
    monkeypatch.setattr(clouds.RunPod, 'get_credential_file_mounts',
                        lambda self: {'~/.runpod/config.toml': str(key_file)})
    remote = '~/.runpod/config.toml'
    assert remote in sky_check.get_cloud_credential_file_mounts(None)
    with skypilot_config.override_skypilot_config(
        {'runpod': {
            'remote_identity': 'NO_UPLOAD'
        }}):
        excluded = _no_upload_clouds('US')
    assert any(isinstance(cloud, clouds.RunPod) for cloud in excluded)
    assert remote not in sky_check.get_cloud_credential_file_mounts(excluded)


STORAGE_FILES = {
    'cloudflare': '~/.cloudflare/r2.credentials',
    'coreweave': '~/.coreweave/cw.credentials',
    'vastdata': '~/.vastdata/vastdata.credentials',
    'huggingface': '~/.cache/huggingface/token',
}


@pytest.fixture(name='storage_credentials')
def all_storage_credentials_present(monkeypatch):
    for name, remote in STORAGE_FILES.items():
        adaptor = getattr(sky_check, name)
        monkeypatch.setattr(adaptor, 'check_storage_credentials', lambda:
                            (True, None))
        monkeypatch.setattr(adaptor,
                            'get_credential_file_mounts',
                            lambda remote=remote: {remote: remote})


def test_storage_credential_config_schema():
    # SkyPilot's validator implements case_insensitive_enum.
    for value in ([], ['huggingface'], ['Cloudflare', 'vastdata']):
        common_utils.validate_schema(
            {'provision': {
                'storage_credentials': value
            }}, schemas.get_config_schema())
    for value in (['s3'], ['huggingface', 'huggingface'], 'cloudflare'):
        with pytest.raises(ValueError):
            common_utils.validate_schema(
                {'provision': {
                    'storage_credentials': value
                }}, schemas.get_config_schema())


@pytest.mark.parametrize('allowed,expected', [
    (None, set(STORAGE_FILES)),
    ([], set()),
    (['huggingface'], {'huggingface'}),
    (['Cloudflare'], {'cloudflare'}),
])
def test_storage_credentials_upload_only_where_allowed(storage_credentials,
                                                       allowed, expected):
    del storage_credentials
    mounts = sky_check.get_cloud_credential_file_mounts(
        set(registry.CLOUD_REGISTRY.values()), storage_credentials=allowed)
    assert set(mounts) == {STORAGE_FILES[name] for name in expected}


@pytest.mark.parametrize('config,expected', [
    ({}, None),
    ({
        'provision': {
            'storage_credentials': []
        }
    }, []),
    ({
        'provision': {
            'storage_credentials': ['huggingface']
        }
    }, ['huggingface']),
])
def test_cluster_config_applies_storage_credential_policy(config, expected):
    patches = [
        mock.patch.object(skypilot_config, '_global_config_context',
                          skypilot_config.ConfigContext()),
        mock.patch('sky.catalog.instance_type_exists', return_value=True),
        mock.patch('sky.catalog.get_accelerators_from_instance_type',
                   return_value={'fake-acc': 2}),
        mock.patch('sky.catalog.get_image_id_from_tag',
                   return_value='fake-image'),
        mock.patch.object(clouds.aws, 'DEFAULT_SECURITY_GROUP_NAME',
                          'fake-default-sg'),
        mock.patch('sky.catalog.get_arch_from_instance_type',
                   return_value='fake-arch'),
        mock.patch(
            'sky.backends.backend_utils._get_yaml_path_from_cluster_name',
            return_value='/tmp/fake/path'),
        mock.patch(
            'sky.backends.backend_utils._deterministic_cluster_yaml_hash',
            return_value='fake-hash'),
        mock.patch('sky.utils.common_utils.fill_template'),
    ]
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        credential_mounts = stack.enter_context(
            mock.patch('sky.check.get_cloud_credential_file_mounts',
                       return_value={}))
        stack.enter_context(skypilot_config.override_skypilot_config(config))
        backend_utils.write_cluster_config(
            to_provision=resources_lib.Resources(cloud=clouds.AWS(),
                                                 instance_type='fake-type: 3'),
            num_nodes=1,
            cluster_config_template='aws-ray.yml.j2',
            cluster_name='display',
            local_wheel_path=pathlib.Path('/tmp/fake'),
            wheel_hash='b1bd84059bc0342f7843fcbe04ab563e',
            region=clouds.Region(name='fake-region'),
            zones=[clouds.Zone(name='fake-zone')],
            dryrun=True,
            keep_launch_fields_in_existing_config=True)
    assert credential_mounts.call_args.kwargs['storage_credentials'] == expected

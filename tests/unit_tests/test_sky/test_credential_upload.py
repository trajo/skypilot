"""Local credential files must be excludable from every cluster."""
import jsonschema
import pytest

from sky import check as sky_check
from sky import clouds
from sky import skypilot_config
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

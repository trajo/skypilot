"""RunPod CUDA version allowlist: resources YAML to utils.launch."""
import types
from unittest import mock

import pytest
import yaml

from sky import resources as resources_lib
from sky.clouds import runpod
from sky.provision.runpod import instance
from sky.utils import common_utils
from sky.utils import resources_utils


def _schema_check(accelerator_args):
    resources, = resources_lib.Resources.from_yaml_config(
        {'accelerator_args': accelerator_args})
    return resources


def test_schema_accepts_versions():
    resources = _schema_check({'runpod_cuda_versions': ['12.8', '13.0']})
    assert resources.accelerator_args == {
        'runpod_cuda_versions': ['12.8', '13.0']
    }


@pytest.mark.parametrize('versions', [[], ['13'], ['13.0.1'], ['13.0', '13.0'],
                                      [13.0], '13.0'])
def test_schema_rejects_invalid_versions(versions):
    with pytest.raises(ValueError, match='Invalid resources YAML'):
        _schema_check({'runpod_cuda_versions': versions})


def _deploy_variables(accelerator_args):
    resources = types.SimpleNamespace(instance_type='1x_RTX4090_SECURE',
                                      image_id=None,
                                      use_spot=False,
                                      docker_username_for_runpod=None,
                                      accelerator_args=accelerator_args)
    resources.assert_launchable = lambda: resources
    region = mock.MagicMock()
    region.name = 'US'
    with mock.patch.object(runpod.RunPod,
                           'get_accelerators_from_instance_type',
                           return_value={'RTX4090': 1}), \
            mock.patch.object(runpod.RunPod,
                              'instance_type_to_hourly_cost',
                              return_value=0.89):
        return runpod.RunPod().make_deploy_resources_variables(
            resources=resources,
            cluster_name=resources_utils.ClusterName('c', 'c'),
            region=region,
            zones=[],
            num_nodes=1)


def _provider(variables, tmp_path):
    output = tmp_path / 'cluster.yml'
    common_utils.fill_template('runpod-ray.yml.j2', {
        'cluster_name_on_cloud': 'c',
        'num_nodes': 1,
        'docker_login_config': None,
        'credentials': {},
        **variables
    }, str(output))
    return yaml.safe_load(output.read_text())['provider']


def test_template_renders_versions(tmp_path):
    variables = _deploy_variables({'runpod_cuda_versions': ['12.8', '13.0']})
    assert variables['runpod_cuda_versions'] == ['12.8', '13.0']
    provider = _provider(variables, tmp_path)
    assert provider['allowed_cuda_versions'] == ['12.8', '13.0']


@pytest.mark.parametrize('accelerator_args', [None, {}])
def test_template_omits_versions_when_unset(accelerator_args, tmp_path):
    variables = _deploy_variables(accelerator_args)
    assert variables['runpod_cuda_versions'] == []
    assert 'allowed_cuda_versions' not in _provider(variables, tmp_path)


@pytest.mark.parametrize('versions', [['12.8', '13.0'], None])
def test_run_instances_passes_versions_to_launch(versions):
    provider_config = {'availability_zone': 'US'}
    if versions is not None:
        provider_config['allowed_cuda_versions'] = versions
    config = mock.MagicMock(count=1,
                            provider_config=provider_config,
                            node_config={
                                'InstanceType': '1x_RTX4090_SECURE',
                                'DiskSize': 100,
                                'ImageId': 'runpod/base',
                                'PublicKey': 'ssh-ed25519 key',
                                'Preemptible': False,
                                'BidPerGPU': 0.89
                            })
    pods = {}

    def launch(**kwargs):
        pods['pod-1'] = {
            'name': 'c-head',
            'status': 'RUNNING',
            'ssh_port': 22
        }
        return 'pod-1'

    with mock.patch.object(instance.utils, 'list_instances',
                           side_effect=lambda: dict(pods)), \
            mock.patch.object(instance.utils, 'launch',
                              side_effect=launch) as launched:
        instance.run_instances('US', 'c', 'c', config)
    assert launched.call_args.kwargs['allowed_cuda_versions'] == versions

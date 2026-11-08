"""Tests for task-level Nebius keys: spot pricing policy and filesystems.

`nebius.spot_pricing_policy_id` makes a spot VM reference a billing pricing
policy (its maximum price) instead of following the spot price. Task-level
`nebius.filesystems` lets one task attach shared filesystems without the
server-wide `nebius.region_configs`. Both are task-overridable, so nothing
changes for other tasks on the same API server.
"""
# pylint: disable=protected-access,redefined-outer-name
import importlib.util
import pathlib
from unittest import mock

import jinja2
import jsonschema
import pytest
import yaml

import sky
from sky import exceptions
from sky import resources as resources_lib
from sky import skypilot_config
from sky.clouds import nebius as nebius_cloud
from sky.provision.nebius import instance
from sky.provision.nebius import utils
from sky.skylet import constants
from sky.utils import resources_utils
from sky.utils import schemas
from sky.utils import validator

_HAS_NEBIUS_SDK = importlib.util.find_spec('nebius') is not None
nebius_sdk_required = pytest.mark.skipif(
    not _HAS_NEBIUS_SDK, reason='Nebius SDK not installed in this environment')

POLICY = 'pricingpolicy-e00example'
FS = 'computefilesystem-e00example'

# ------------------------------------------------------------------ helpers


def _variables(overrides=None, use_spot=True, region_name='eu-north1'):
    resources = resources_lib.Resources(cloud=nebius_cloud.Nebius(),
                                        instance_type='cpu-d3_4vcpu-16gb',
                                        region=region_name,
                                        use_spot=use_spot,
                                        _cluster_config_overrides=overrides or
                                        {}).assert_launchable()
    region = mock.MagicMock()
    region.name = region_name
    with mock.patch.object(utils,
                           'get_project_by_region',
                           return_value='project-id'):
        return nebius_cloud.Nebius().make_deploy_resources_variables(
            resources,
            resources_utils.ClusterName(display_name='c',
                                        name_on_cloud='c-1234'), region, None,
            1)


def _render(variables):
    path = pathlib.Path(sky.__file__).parent / 'templates' / 'nebius-ray.yml.j2'
    defaults = dict(num_nodes=1,
                    credentials={},
                    initial_setup_commands=[],
                    docker_image=None,
                    ssh_private_key='~/.ssh/key',
                    ssh_proxy_command=None,
                    cluster_name_on_cloud='c-1234',
                    ssh_public_key='ssh-ed25519 AAAA',
                    disk_size=100)
    defaults.update(variables)
    return yaml.safe_load(jinja2.Template(path.read_text()).render(**defaults))


def _node_config(variables):
    return _render(
        variables)['available_node_types']['ray_head_default']['node_config']


def _validate(config):
    # SkyPilot's validator: it knows custom keywords like case_sensitive_enum.
    validator.get_schema_validator()(
        schemas.get_config_schema()).validate(config)


def _validate_task_config(config):
    return sky.Task.from_yaml_config({
        'resources': {
            'infra': 'nebius'
        },
        'config': config,
    })


# ------------------------------------------------------------ pricing policy


def test_pricing_policy_is_task_overridable():
    assert (
        'nebius',
        'spot_pricing_policy_id') in constants.OVERRIDEABLE_CONFIG_KEYS_IN_TASK
    task = _validate_task_config({'nebius': {'spot_pricing_policy_id': POLICY}})
    resources = next(iter(task.resources))
    assert resources.cluster_config_overrides['nebius'][
        'spot_pricing_policy_id'] == POLICY


@pytest.mark.parametrize('value', ['', 3, None, ['x']])
def test_pricing_policy_schema_rejects_other_values(value):
    with pytest.raises(jsonschema.ValidationError):
        _validate({'nebius': {'spot_pricing_policy_id': value}})


def test_pricing_policy_reaches_node_config_for_spot():
    variables = _variables({'nebius': {'spot_pricing_policy_id': POLICY}})
    assert variables['spot_pricing_policy_id'] == POLICY
    assert _node_config(variables)['spot_pricing_policy_id'] == POLICY


def test_no_pricing_policy_by_default():
    variables = _variables()
    assert variables['spot_pricing_policy_id'] is None
    assert 'spot_pricing_policy_id' not in _node_config(variables)


def test_pricing_policy_ignored_for_on_demand():
    variables = _variables({'nebius': {
        'spot_pricing_policy_id': POLICY
    }},
                           use_spot=False)
    assert variables['spot_pricing_policy_id'] is None
    assert 'spot_pricing_policy_id' not in _node_config(variables)


def test_pricing_policy_from_server_config():
    # A task override is the intended use; a server-wide value also works,
    # like every other task-overridable key, and the task value wins.
    with skypilot_config.override_skypilot_config(
        {'nebius': {
            'spot_pricing_policy_id': POLICY
        }}):
        assert _variables()['spot_pricing_policy_id'] == POLICY
        assert _variables(
            {'nebius': {
                'spot_pricing_policy_id': 'pricingpolicy-task'
            }})['spot_pricing_policy_id'] == 'pricingpolicy-task'
    assert _variables()['spot_pricing_policy_id'] is None


@pytest.mark.parametrize('policy', [None, POLICY])
def test_run_instances_passes_pricing_policy(monkeypatch, tmp_path, policy):
    # pylint: disable=import-outside-toplevel
    from sky.provision import common
    from sky.provision.nebius import journal
    monkeypatch.setattr(journal, 'ROOT', tmp_path)
    seen = []

    def fake_launch(creation, **kwargs):
        seen.append(kwargs)
        creation.update(accepted=True, id='computeinstance-0', name='n')
        return 'computeinstance-0'

    monkeypatch.setattr(utils, 'launch', fake_launch)
    monkeypatch.setattr(utils, 'list_instances', lambda project_id: {})
    node_config = {
        'InstanceType': 'cpu-d3_4vcpu-16gb',
        'ImageId': 'image',
        'DiskSize': 100,
        'UserData': '',
        'use_spot': True,
        'disk_tier': 'medium',
    }
    if policy is not None:
        node_config['spot_pricing_policy_id'] = policy
    config = common.ProvisionConfig(provider_config={
        'region': 'eu-north1',
        'project_id': 'project-id',
        'use_internal_ips': False
    },
                                    authentication_config={},
                                    docker_config={},
                                    node_config=node_config,
                                    count=1,
                                    tags={},
                                    resume_stopped_nodes=True,
                                    ports_to_open_on_launch=None)
    instance.run_instances('eu-north1', 'display', 'cloud', config)
    assert [k['spot_pricing_policy_id'] for k in seen] == [policy]


@pytest.fixture
def instance_service(monkeypatch):
    # pylint: disable=import-outside-toplevel
    from sky.adaptors import nebius
    service = mock.Mock()
    monkeypatch.setattr(nebius, 'sdk', lambda: None)
    monkeypatch.setattr(nebius, 'sync_call', lambda response: response)
    monkeypatch.setattr(nebius.compute(), 'InstanceServiceClient',
                        lambda _: service)
    monkeypatch.setattr(utils, 'get_project_by_region', lambda _: 'project-id')
    monkeypatch.setattr(utils, 'get_subnet_id', lambda *_: 'subnet-id')
    compute = nebius.compute()
    service.get_by_name.return_value = compute.Instance(
        metadata=nebius.nebius_common().ResourceMetadata(id='instance-id'),
        status=compute.InstanceStatus(
            state=compute.InstanceStatus.InstanceState.STARTING))
    return service


def _launch_spec(service, **kwargs):
    utils.launch(cluster_name_on_cloud='cloud',
                 node_type='head',
                 platform='cpu-d3',
                 preset='4vcpu-16gb',
                 region='eu-north1',
                 image_id_or_family='computeimage-test',
                 disk_size=100,
                 user_data='',
                 associate_public_ip_address=False,
                 filesystems=[],
                 disk_tier=resources_utils.DiskTier.MEDIUM,
                 creation={},
                 **kwargs)
    return service.create.call_args.args[0].spec


@nebius_sdk_required
def test_launch_uses_pricing_policy_instead_of_spot_price(instance_service):
    spec = _launch_spec(instance_service,
                        use_spot=True,
                        spot_pricing_policy_id=POLICY)
    assert spec.which_field_in_oneof('pricing_model') == 'spot_pricing_policy'
    assert spec.spot_pricing_policy.id == POLICY
    assert spec.check_presence('preemptible')


@nebius_sdk_required
def test_launch_follows_spot_price_without_policy(instance_service):
    spec = _launch_spec(instance_service, use_spot=True)
    assert spec.which_field_in_oneof('pricing_model') == 'follows_spot_price'


@nebius_sdk_required
def test_launch_on_demand_ignores_policy(instance_service):
    spec = _launch_spec(instance_service,
                        use_spot=False,
                        spot_pricing_policy_id=POLICY)
    assert spec.which_field_in_oneof('pricing_model') is None
    assert not spec.check_presence('preemptible')


# --------------------------------------------------------------- filesystems


def test_filesystems_are_task_overridable():
    assert ('nebius',
            'filesystems') in constants.OVERRIDEABLE_CONFIG_KEYS_IN_TASK
    entry = {
        'filesystem_id': FS,
        'mount_path': '/mnt/weights',
        'attach_mode': 'READ_ONLY',
        'region': 'eu-north1'
    }
    task = _validate_task_config({'nebius': {'filesystems': [entry]}})
    resources = next(iter(task.resources))
    assert resources.cluster_config_overrides['nebius']['filesystems'] == [
        entry
    ]


@pytest.mark.parametrize('entry', [
    {
        'mount_path': '/mnt/x'
    },
    {
        'filesystem_id': FS,
        'attach_mode': 'WRITE'
    },
    {
        'filesystem_id': FS,
        'mount_path': 'relative/path'
    },
    {
        'filesystem_id': FS,
        'unknown': 1
    },
])
def test_filesystems_schema_rejects_bad_entries(entry):
    with pytest.raises(jsonschema.ValidationError):
        _validate({'nebius': {'filesystems': [entry]}})


def _fs(variables):
    return [(f['filesystem_id'], f['filesystem_mount_path'],
             f['filesystem_attach_mode'], f['filesystem_mount_tag'])
            for f in variables['filesystems']]


def test_task_filesystem_is_attached():
    variables = _variables({
        'nebius': {
            'filesystems': [{
                'filesystem_id': FS,
                'mount_path': '/mnt/weights',
                'attach_mode': 'READ_ONLY',
            }]
        }
    })
    assert _fs(variables) == [(FS, '/mnt/weights', 'READ_ONLY',
                               'filesystem-skypilot-1')]
    node = _node_config(variables)
    assert node['filesystems'] == [{
        'filesystem_id': FS,
        'filesystem_mount_tag': 'filesystem-skypilot-1',
        'filesystem_attach_mode': 'READ_ONLY',
        'filesystem_mount_path': '/mnt/weights',
    }]


def test_task_filesystem_defaults():
    variables = _variables({'nebius': {'filesystems': [{'filesystem_id': FS}]}})
    assert _fs(variables) == [(FS, '/mnt/filesystem-skypilot-1', 'READ_WRITE',
                               'filesystem-skypilot-1')]


def test_task_filesystem_for_its_region_only():
    overrides = {
        'nebius': {
            'filesystems': [
                {
                    'filesystem_id': FS,
                    'region': 'eu-north1',
                    'mount_path': '/mnt/a'
                },
                {
                    'filesystem_id': 'computefilesystem-other',
                    'region': 'us-central1',
                    'mount_path': '/mnt/a'
                },
            ]
        }
    }
    assert _fs(_variables(overrides)) == [(FS, '/mnt/a', 'READ_WRITE',
                                           'filesystem-skypilot-1')]
    assert _fs(_variables(overrides, region_name='us-central1')) == [
        ('computefilesystem-other', '/mnt/a', 'READ_WRITE',
         'filesystem-skypilot-1')
    ]


def test_region_without_task_filesystem_is_skipped():
    # Launching without the declared mount would let the job write to the boot
    # disk, so failover skips regions where none of the task's filesystems
    # exist.
    overrides = {
        'nebius': {
            'filesystems': [{
                'filesystem_id': FS,
                'region': 'eu-north1'
            }]
        }
    }
    with pytest.raises(exceptions.ResourcesUnavailableError,
                       match='filesystem'):
        _variables(overrides, region_name='us-central1')


def test_task_filesystems_follow_server_region_filesystems():
    server = {
        'nebius': {
            'region_configs': {
                'eu-north1': {
                    'filesystems': [{
                        'filesystem_id': 'computefilesystem-server',
                        'mount_path': '/mnt/shared',
                    }]
                }
            }
        }
    }
    task = {
        'nebius': {
            'filesystems': [{
                'filesystem_id': FS,
                'mount_path': '/mnt/weights'
            }]
        }
    }
    with skypilot_config.override_skypilot_config(server):
        assert _fs(_variables(task)) == [
            ('computefilesystem-server', '/mnt/shared', 'READ_WRITE',
             'filesystem-skypilot-1'),
            (FS, '/mnt/weights', 'READ_WRITE', 'filesystem-skypilot-2'),
        ]
        # Unchanged for tasks that declare none.
        assert _fs(_variables()) == [('computefilesystem-server', '/mnt/shared',
                                      'READ_WRITE', 'filesystem-skypilot-1')]


@pytest.mark.parametrize('second', [
    {
        'filesystem_id': FS,
        'mount_path': '/mnt/b'
    },
    {
        'filesystem_id': 'computefilesystem-other',
        'mount_path': '/mnt/a'
    },
])
def test_duplicate_filesystem_or_mount_path_is_an_error(second):
    overrides = {
        'nebius': {
            'filesystems': [{
                'filesystem_id': FS,
                'mount_path': '/mnt/a'
            }, second]
        }
    }
    with pytest.raises(ValueError, match='more than once'):
        _variables(overrides)


# ------------------------------------------------- mount after restart in place

WEIGHTS = {
    'nebius': {
        'filesystems': [{
            'filesystem_id': FS,
            'mount_path': '/mnt/weights'
        }]
    }
}


def _setup_script(variables):
    return '\n'.join(_render(variables)['setup_commands'])


def _cloud_init(variables):
    return yaml.safe_load(_node_config(variables)['UserData'])


def test_host_mounts_filesystems_at_every_boot():
    # The setup stage is cached per instance ID on the API server, so a VM
    # restarted in place (FAILOVER after a spot preemption, or `sky start`)
    # skips it; with a docker image it also runs inside the container. The
    # host therefore mounts from cloud-init bootcmd, which runs at every boot
    # before sshd (so before SkyPilot starts the container).
    cloud_init = _cloud_init(_variables(WEIGHTS))
    assert cloud_init['bootcmd'] == [
        'mkdir -p /mnt/weights',
        'mountpoint -q /mnt/weights || '
        'mount -t virtiofs filesystem-skypilot-1 /mnt/weights',
        'chmod a+w /mnt/weights || true',
    ]
    # The SSH user is still set up.
    assert cloud_init['users'][0]['name'] == 'skypilot:ssh_user'


def test_setup_mount_is_idempotent():
    script = _setup_script(_variables(WEIGHTS))
    assert 'sudo mkdir -p /mnt/weights' in script
    assert ('mountpoint -q /mnt/weights || sudo mount -t virtiofs '
            'filesystem-skypilot-1 /mnt/weights') in script


def test_docker_container_binds_the_host_mounts():
    resources = resources_lib.Resources(
        cloud=nebius_cloud.Nebius(),
        instance_type='cpu-d3_4vcpu-16gb',
        region='eu-north1',
        image_id='docker:vllm/vllm-openai:latest',
        _cluster_config_overrides=WEIGHTS).assert_launchable()
    region = mock.MagicMock()
    region.name = 'eu-north1'
    with mock.patch.object(utils,
                           'get_project_by_region',
                           return_value='project-id'):
        variables = nebius_cloud.Nebius().make_deploy_resources_variables(
            resources,
            resources_utils.ClusterName(display_name='c',
                                        name_on_cloud='c-1234'), region, None,
            1)
    assert '-v /mnt/weights:/mnt/weights' in variables['docker_run_options']


def test_nothing_mounted_without_filesystems():
    variables = _variables()
    assert 'bootcmd' not in _cloud_init(variables)
    assert 'virtiofs' not in _setup_script(variables)
    assert 'docker_run_options' not in variables

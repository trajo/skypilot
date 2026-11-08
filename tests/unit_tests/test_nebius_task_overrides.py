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


def _host_commands(variables):
    return _render(variables).get('host_setup_commands') or []


def _docker_variables(overrides):
    resources = resources_lib.Resources(
        cloud=nebius_cloud.Nebius(),
        instance_type='cpu-d3_4vcpu-16gb',
        region='eu-north1',
        image_id='docker:vllm/vllm-openai:latest',
        _cluster_config_overrides=overrides).assert_launchable()
    region = mock.MagicMock()
    region.name = 'eu-north1'
    with mock.patch.object(utils,
                           'get_project_by_region',
                           return_value='project-id'):
        return nebius_cloud.Nebius().make_deploy_resources_variables(
            resources,
            resources_utils.ClusterName(display_name='c',
                                        name_on_cloud='c-1234'), region, None,
            1)


def test_filesystems_mounted_only_by_host_setup_commands():
    # One authoritative path: the provisioner runs host_setup_commands on the
    # host at every provision (uncached), before docker. Neither cloud-init
    # nor the cached, possibly in-container setup stage mounts.
    variables = _variables(WEIGHTS)
    commands = _host_commands(variables)
    assert len(commands) == 1 and 'virtiofs' in commands[0]
    rendered = _render(variables)
    assert 'virtiofs' not in '\n'.join(rendered['setup_commands'])
    node = rendered['available_node_types']['ray_head_default']['node_config']
    assert 'bootcmd' not in node['UserData']


def test_nothing_mounted_without_filesystems():
    variables = _variables()
    assert _host_commands(variables) == []
    assert 'virtiofs' not in yaml.safe_dump(_render(variables))
    assert 'docker_run_options' not in variables


@pytest.mark.parametrize('mode,suffix', [('READ_WRITE', ''),
                                         ('READ_ONLY', ':ro')])
def test_docker_container_binds_the_host_mounts(mode, suffix):
    overrides = {
        'nebius': {
            'filesystems': [{
                'filesystem_id': FS,
                'mount_path': '/mnt/weights',
                'attach_mode': mode
            }]
        }
    }
    assert (f'-v /mnt/weights:/mnt/weights{suffix}'
            in _docker_variables(overrides)['docker_run_options'])


SHIM = """#!/bin/sh
echo "$(basename "$0") $*" >> "$SHIM_LOG"
case "$(basename "$0")" in
  sudo) exec "$@" ;;
  mountpoint) [ -n "$MOUNTED" ] ;;
  mount) [ -z "$MOUNT_FAILS" ] ;;
  findmnt) [ -n "$MOUNTED$MOUNT_OK" ] && echo "$FINDMNT" ;;
  *) true ;;
esac
"""


def _run_host_command(tmp_path, attach_mode='READ_WRITE', **env):
    import subprocess  # pylint: disable=import-outside-toplevel
    overrides = {
        'nebius': {
            'filesystems': [{
                'filesystem_id': FS,
                'mount_path': '/mnt/weights',
                'attach_mode': attach_mode
            }]
        }
    }
    (command,) = _host_commands(_variables(overrides))
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('sudo', 'mkdir', 'mountpoint', 'mount', 'findmnt', 'chmod'):
        shim = bin_dir / name
        shim.write_text(SHIM)
        shim.chmod(0o755)
    log = tmp_path / 'log'
    log.write_text('')
    full_env = {
        'PATH': f'{bin_dir}:/usr/bin:/bin',
        'SHIM_LOG': str(log),
        'FINDMNT': 'virtiofs filesystem-skypilot-1',
        **env
    }
    result = subprocess.run(['bash', '-c', command],
                            env=full_env,
                            capture_output=True,
                            text=True,
                            check=False)
    calls = [line.split()[0] for line in log.read_text().splitlines()]
    return result.returncode, calls, result.stderr


def test_host_mount_when_not_mounted(tmp_path):
    rc, calls, _ = _run_host_command(tmp_path, MOUNT_OK='1')
    assert rc == 0
    assert 'mount' in calls and 'chmod' in calls


def test_host_mount_skips_an_existing_mount(tmp_path):
    rc, calls, _ = _run_host_command(tmp_path, MOUNTED='1')
    assert rc == 0
    assert 'mount' not in calls


def test_host_mount_failure_fails_provisioning(tmp_path):
    rc, calls, stderr = _run_host_command(tmp_path, MOUNT_FAILS='1')
    assert rc != 0
    assert 'chmod' not in calls
    assert '/mnt/weights' in stderr


def test_host_mount_of_another_filesystem_fails(tmp_path):
    rc, _, stderr = _run_host_command(tmp_path,
                                      MOUNTED='1',
                                      FINDMNT='ext4 /dev/vda1')
    assert rc != 0
    assert 'filesystem-skypilot-1' in stderr


def test_read_only_mount_is_not_chmodded(tmp_path):
    rc, calls, _ = _run_host_command(tmp_path,
                                     attach_mode='READ_ONLY',
                                     MOUNT_OK='1')
    assert rc == 0
    assert 'chmod' not in calls


# ------------------------------------------------------ mount path validation


@pytest.mark.parametrize('path', [
    '/mnt/model weights', '/mnt/w:x', '/mnt/$(id)', "/mnt/a'b", '/mnt/../etc',
    '/', '/mnt/a\nb'
])
def test_unsafe_mount_paths_are_rejected(path):
    with pytest.raises(jsonschema.ValidationError):
        _validate({
            'nebius': {
                'filesystems': [{
                    'filesystem_id': FS,
                    'mount_path': path
                }]
            }
        })
    # Server region entries are checked when used, too.
    server = {
        'nebius': {
            'region_configs': {
                'eu-north1': {
                    'filesystems': [{
                        'filesystem_id': FS,
                        'mount_path': path
                    }]
                }
            }
        }
    }
    with skypilot_config.override_skypilot_config(server):
        with pytest.raises(ValueError, match='mount path'):
            _variables()


@pytest.mark.parametrize('fs_id', ['', 'a b', 'fs:1', 'fs$(id)'])
def test_unsafe_filesystem_ids_are_rejected(fs_id):
    with pytest.raises(jsonschema.ValidationError):
        _validate({'nebius': {'filesystems': [{'filesystem_id': fs_id}]}})


def test_mount_paths_are_normalized():
    overrides = {
        'nebius': {
            'filesystems': [{
                'filesystem_id': FS,
                'mount_path': '/mnt//weights/'
            }]
        }
    }
    assert _fs(_variables(overrides))[0][1] == '/mnt/weights'


@pytest.mark.parametrize('second', ['/mnt/a/', '/mnt/a/b', '/mnt'])
def test_equal_or_nested_mount_paths_are_rejected(second):
    overrides = {
        'nebius': {
            'filesystems': [{
                'filesystem_id': FS,
                'mount_path': '/mnt/a'
            }, {
                'filesystem_id': 'computefilesystem-other',
                'mount_path': second
            }]
        }
    }
    with pytest.raises(ValueError, match='mount path'):
        _variables(overrides)


# ------------------------------------------------------ provisioner host hook


def _runner(returncode=0):
    runner = mock.Mock()
    runner.run.return_value = (returncode, 'out', 'err')
    return runner


def _cluster_info(n):
    # pylint: disable=import-outside-toplevel
    from sky.provision import common
    instances = {
        f'i-{k}': [common.InstanceInfo(f'i-{k}', f'10.0.0.{k}', None, {})]
        for k in range(n)
    }
    return common.ClusterInfo(instances=instances,
                              head_instance_id='i-0',
                              provider_name='nebius',
                              provider_config={})


def test_host_setup_commands_run_on_every_node_every_time():
    # pylint: disable=import-outside-toplevel
    from sky import provision
    from sky.provision import instance_setup
    runners = [_runner(), _runner()]
    with mock.patch.object(provision,
                           'get_command_runners',
                           return_value=runners):
        for _ in range(2):  # No caching: a restart in place reruns them.
            instance_setup.run_host_setup_commands('c-1234', ['a', 'b'],
                                                   _cluster_info(2), {})
    for runner in runners:
        assert [c.args[0] for c in runner.run.call_args_list
               ] == ['a', 'b', 'a', 'b']


def test_host_setup_command_failure_raises():
    # pylint: disable=import-outside-toplevel
    from sky import provision
    from sky.provision import instance_setup
    with mock.patch.object(provision,
                           'get_command_runners',
                           return_value=[_runner(1)]):
        with pytest.raises(RuntimeError, match='err'):
            instance_setup.run_host_setup_commands('c-1234', ['a'],
                                                   _cluster_info(1), {})


def test_provisioner_runs_host_commands_before_docker(monkeypatch):
    # pylint: disable=import-outside-toplevel
    from sky.provision import common
    from sky.provision import instance_setup
    from sky.provision import provisioner
    order = []

    class Stop(Exception):
        pass

    def initialize_docker(*args, **kwargs):
        del args, kwargs
        order.append('docker')
        raise Stop()

    def run_host_setup_commands(cluster_name, commands, cluster_info,
                                ssh_credentials):
        del cluster_name, cluster_info
        assert 'docker_user' not in ssh_credentials  # On the host.
        order.append(('host', tuple(commands)))

    config = {
        'provider': {},
        'docker': {
            'image': 'img'
        },
        'host_setup_commands': ['mount it'],
        'setup_commands': []
    }
    monkeypatch.setattr(provisioner.global_user_state, 'get_cluster_yaml_dict',
                        lambda _: config)
    monkeypatch.setattr(provisioner.global_user_state,
                        'get_handle_from_cluster_name', lambda _: mock.Mock())
    monkeypatch.setattr(provisioner.global_user_state, 'update_cluster_handle',
                        lambda *_: None)
    monkeypatch.setattr(provisioner.provision, 'get_cluster_info',
                        lambda *a, **k: _cluster_info(1))
    monkeypatch.setattr(provisioner.backend_utils, 'ssh_credential_from_yaml',
                        lambda *a, **k: {'ssh_user': 'ubuntu'})
    monkeypatch.setattr(provisioner, 'wait_for_ssh',
                        lambda *a: order.append('ssh'))
    monkeypatch.setattr(instance_setup, 'run_host_setup_commands',
                        run_host_setup_commands)
    monkeypatch.setattr(instance_setup, 'initialize_docker', initialize_docker)
    resources = mock.Mock()
    resources.cloud = nebius_cloud.Nebius()
    record = common.ProvisionRecord(provider_name='nebius',
                                    region='eu-north1',
                                    zone=None,
                                    cluster_name='c-1234',
                                    head_instance_id='i-0',
                                    resumed_instance_ids=['i-0'],
                                    created_instance_ids=[])
    with pytest.raises(Stop):
        provisioner._post_provision_setup(
            resources,
            resources_utils.ClusterName(display_name='c',
                                        name_on_cloud='c-1234'), 'yaml', record,
            None)
    assert order == ['ssh', ('host', ('mount it',)), 'docker']

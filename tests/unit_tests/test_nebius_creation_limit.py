"""Tests for the Nebius bound on instance creations per cluster.

`nebius.max_instance_creations` caps how many VMs a cluster may ever create,
across failover candidates and managed-job recoveries, using the creation
journal shared with Vast (sky.provision.creation_journal).
"""
# pylint: disable=protected-access
import importlib.util
import pathlib
import threading
from unittest import mock

import jinja2
import jsonschema
import pytest

import sky
from sky import exceptions
from sky import resources as resources_lib
from sky.clouds import nebius as nebius_cloud
from sky.provision import common
from sky.provision.nebius import instance
from sky.provision.nebius import journal
from sky.provision.nebius import utils
from sky.skylet import constants
from sky.utils import resources_utils
from sky.utils import schemas

_HAS_NEBIUS_SDK = importlib.util.find_spec('nebius') is not None
nebius_sdk_required = pytest.mark.skipif(
    not _HAS_NEBIUS_SDK, reason='Nebius SDK not installed in this environment')

CLUSTER = 'cloud'


@pytest.fixture(autouse=True)
def isolated_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, 'ROOT', tmp_path)


def _config(count=1, limit=2):
    provider_config = {
        'region': 'eu-north1',
        'project_id': 'project-id',
        'use_internal_ips': False
    }
    if limit is not None:
        provider_config['max_instance_creations'] = limit
    return common.ProvisionConfig(provider_config=provider_config,
                                  authentication_config={},
                                  docker_config={},
                                  node_config={
                                      'InstanceType': 'cpu-d3_4vcpu-16gb',
                                      'ImageId': 'image',
                                      'DiskSize': 100,
                                      'UserData': '',
                                      'use_spot': True,
                                      'disk_tier': 'medium',
                                  },
                                  count=count,
                                  tags={},
                                  resume_stopped_nodes=True,
                                  ports_to_open_on_launch=None)


class FakeCloud:
    """Nebius at the utils boundary: instance listing, start and launch.

    `outcomes` scripts the creates in call order. Each outcome mirrors what
    utils.launch notes in `creation` before raising or returning.
    """

    def __init__(self, monkeypatch):
        self.instances = {}
        self.started = []
        self.launched = []
        self.pending_at_launch = []
        self.outcomes = []
        self.running = 0
        self._lock = threading.Lock()
        monkeypatch.setattr(utils, 'list_instances',
                            lambda project_id: dict(self.instances))
        monkeypatch.setattr(utils, 'start', self.started.append)
        monkeypatch.setattr(utils, 'launch', self.launch)
        monkeypatch.setattr(instance.time, 'sleep', lambda _: None)

    def launch(self, cluster_name_on_cloud, node_type, creation, **kwargs):
        with self._lock:
            self.running += 1
        try:
            return self._launch(cluster_name_on_cloud, node_type, creation,
                                **kwargs)
        finally:
            with self._lock:
                self.running -= 1

    def _launch(self, cluster_name_on_cloud, node_type, creation, **kwargs):
        del kwargs
        with self._lock:
            record = journal.read(cluster_name_on_cloud)
            self.pending_at_launch.append(
                len(record['pending'] or []) if record else None)
            index = len(self.launched)
            self.launched.append(node_type)
            outcome = self.outcomes[index] if index < len(
                self.outcomes) else 'ok'
        name = f'{cluster_name_on_cloud}-{index:04x}-{node_type}'
        if outcome == 'before_create':
            raise ValueError('no subnet')
        if outcome == 'slow_create':
            # Not at the create call yet when another worker fails.
            threading.Event().wait(0.5)
            outcome = 'ok'
        creation['name'] = name
        if outcome == 'refused':
            creation['refused'] = True
            raise RuntimeError('RESOURCE_EXHAUSTED: quota')
        if outcome == 'unknown':
            raise TimeoutError('response lost')
        creation['accepted'] = True
        creation['id'] = f'computeinstance-{index}'
        if outcome == 'failed_after_create':
            raise RuntimeError(f'Instance {name} failed to start.')
        if outcome == 'slow':
            # Still polling for the VM while another worker fails.
            threading.Event().wait(0.5)
            outcome = 'ok'
        assert outcome == 'ok', outcome
        return creation['id']


@pytest.fixture
def cloud(monkeypatch):
    return FakeCloud(monkeypatch)


def _run(config):
    return instance.run_instances('eu-north1', 'display', CLUSTER, config)


def _counted(record):
    return list(record['ids'])


# ---------------------------------------------------------------- journal


def test_reserve_and_resolve_counting_rules():
    journal.register('display', CLUSTER)
    slots = journal.reserve(CLUSTER, 4, 5)
    assert len(set(slots)) == 4
    assert sorted(journal.read(CLUSTER)['pending']) == sorted(slots)
    # Confirmed and unknown creates count; a refused one is released.
    journal.resolve(
        CLUSTER, {
            slots[0]: 'computeinstance-1',
            slots[1]: None,
            slots[2]: 'cloud-ab12-worker',
        })
    record = journal.read(CLUSTER)
    assert record['ids'] == ['computeinstance-1', 'cloud-ab12-worker']
    assert record['pending'] == [slots[3]]
    journal.resolve(CLUSTER, {slots[3]: None})
    assert journal.read(CLUSTER)['pending'] is None
    # Nothing is ever deleted: the counted creates bound the next reserve.
    with pytest.raises(exceptions.InstanceCreationLimitError):
        journal.reserve(CLUSTER, 4, 5)
    assert journal.reserve(CLUSTER, 3, 5)


def test_reserve_takes_all_slots_or_none():
    journal.register('display', CLUSTER)
    before = journal.read(CLUSTER)
    with pytest.raises(exceptions.InstanceCreationLimitError,
                       match='Nebius instance creation limit'):
        journal.reserve(CLUSTER, 3, 2)
    assert journal.read(CLUSTER) == before


def test_journal_directory_is_nebius_specific(monkeypatch):
    monkeypatch.undo()
    assert journal.ROOT == pathlib.Path.home() / '.sky' / 'nebius-provisioning'


# ----------------------------------------------------------- run_instances


@pytest.mark.parametrize('outcome,counted', [
    ('ok', ['computeinstance-0']),
    ('failed_after_create', ['computeinstance-0']),
    ('unknown', ['cloud-0000-head']),
    ('refused', []),
    ('before_create', []),
])
def test_each_create_outcome(cloud, outcome, counted):
    cloud.outcomes = [outcome]
    if outcome == 'ok':
        _run(_config())
    else:
        with pytest.raises(Exception):
            _run(_config())
    record = journal.read(CLUSTER)
    assert _counted(record) == counted
    assert record['pending'] is None
    # The slot was durable before the create was attempted.
    assert cloud.pending_at_launch == [1]


def test_restart_in_place_does_not_count(cloud):
    journal.register('display', CLUSTER)
    journal.resolve(CLUSTER,
                    {journal.reserve(CLUSTER, 1, 1)[0]: 'computeinstance-0'})
    before = journal.read(CLUSTER)
    cloud.instances = {
        'computeinstance-0': {
            'name': 'cloud-0000-head',
            'status': 'STOPPED'
        }
    }
    record = _run(_config(limit=1))
    assert cloud.started == ['computeinstance-0']
    assert cloud.launched == []
    assert record.resumed_instance_ids == ['computeinstance-0']
    assert record.created_instance_ids == []
    assert journal.read(CLUSTER) == before


def test_resume_plus_new_worker_counts_only_the_worker(cloud):
    cloud.instances = {
        'computeinstance-9': {
            'name': 'cloud-0009-head',
            'status': 'STOPPED'
        }
    }
    _run(_config(count=2, limit=1))
    assert cloud.started == ['computeinstance-9']
    assert cloud.launched == ['worker']
    assert _counted(journal.read(CLUSTER)) == ['computeinstance-0']


def test_all_creates_reserved_before_the_first(cloud):
    _run(_config(count=3, limit=3))
    assert sorted(cloud.launched) == ['head', 'worker', 'worker']
    # The head was created with every slot of the cluster already reserved.
    assert cloud.pending_at_launch[0] == 3
    record = journal.read(CLUSTER)
    assert len(record['ids']) == 3
    assert record['pending'] is None


@pytest.mark.parametrize('outcome,count', [('refused', 2), ('unknown', 3),
                                           ('failed_after_create', 3),
                                           ('before_create', 2)])
def test_parallel_worker_create_failing(cloud, outcome, count):
    cloud.outcomes = ['ok', 'ok', outcome]
    with pytest.raises(Exception):
        _run(_config(count=3, limit=3))
    record = journal.read(CLUSTER)
    assert len(record['ids']) == count
    assert record['pending'] is None
    # A relaunch of the cluster is bounded by what was counted.
    cloud.instances = {}
    cloud.launched.clear()
    if count == 3:
        with pytest.raises(exceptions.InstanceCreationLimitError):
            _run(_config(count=1, limit=3))
        assert cloud.launched == []
    else:
        _run(_config(count=1, limit=3))
        assert cloud.launched == ['head']


@pytest.mark.parametrize('slow', ['slow', 'slow_create'])
def test_failing_worker_waits_for_the_others(cloud, monkeypatch, slow):
    # A worker that fails must not settle the journal while another worker
    # is still creating: that create would go uncounted (or be released
    # before it ran), and its VM could outlive the teardown.

    def run_in_parallel(func, args):
        # As subprocess_utils.run_in_parallel: results in order, the first
        # error raised as soon as it occurs, other threads left running
        # (ThreadPool.terminate does not join its threads). Threads start in
        # order here so that call order matches argument order.
        results = {}
        errors = {}

        def run(arg):
            try:
                results[arg] = func(arg)
            except Exception as e:  # pylint: disable=broad-except
                errors[arg] = e

        threads = []
        for arg in args:
            started = len(cloud.launched)
            threads.append(threading.Thread(target=run, args=(arg,)))
            threads[-1].start()
            while len(cloud.launched) == started and threads[-1].is_alive():
                threading.Event().wait(0.01)
        for arg, thread in zip(args, threads):
            thread.join()
            if arg in errors:
                raise errors[arg]
        return [results[arg] for arg in args]

    monkeypatch.setattr(instance.subprocess_utils, 'run_in_parallel',
                        run_in_parallel)
    cloud.outcomes = ['ok', 'refused', slow]
    with pytest.raises(RuntimeError, match='RESOURCE_EXHAUSTED'):
        _run(_config(count=3, limit=3))
    assert cloud.running == 0
    record = journal.read(CLUSTER)
    assert len(record['ids']) == 2
    assert record['pending'] is None


def test_limit_reached_raises_before_any_create(cloud):
    journal.register('display', CLUSTER)
    journal.resolve(
        CLUSTER, {
            slot: f'computeinstance-{i}'
            for i, slot in enumerate(journal.reserve(CLUSTER, 2, 2))
        })
    before = journal.read(CLUSTER)
    with pytest.raises(exceptions.InstanceCreationLimitError) as error:
        _run(_config(limit=2))
    assert isinstance(error.value, exceptions.ProvisionUnsupportedError)
    assert not isinstance(error.value, exceptions.ResourcesUnavailableError)
    assert cloud.launched == []
    assert journal.read(CLUSTER) == before


def test_cluster_larger_than_remaining_creations_creates_nothing(cloud):
    with pytest.raises(exceptions.InstanceCreationLimitError):
        _run(_config(count=3, limit=2))
    assert cloud.launched == []
    record = journal.read(CLUSTER)
    assert record['ids'] == [] and record['pending'] is None


def test_unresolved_pending_slot_fails_closed(cloud):
    journal.register('display', CLUSTER)
    journal.reserve(CLUSTER, 1, 3)  # A create interrupted mid-flight.
    with pytest.raises(RuntimeError, match='Unresolved Nebius create'):
        _run(_config(limit=3))
    assert cloud.launched == []
    assert len(journal.read(CLUSTER)['pending']) == 1


def test_identity_mismatch_fails_closed(cloud):
    journal.register('other-display', CLUSTER)
    with pytest.raises(RuntimeError, match='identity mismatch'):
        _run(_config())
    assert cloud.launched == []


def test_without_limit_no_journal_is_written(cloud, tmp_path):
    _run(_config(count=2, limit=None))
    assert sorted(cloud.launched) == ['head', 'worker']
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------- utils.launch


def _request_error(code_name):
    # pylint: disable=import-outside-toplevel
    from grpc import StatusCode
    from nebius.aio import service_error
    return service_error.RequestError(
        service_error.RequestStatusExtended(code=getattr(StatusCode, code_name),
                                            message='m',
                                            details=[],
                                            service_errors=[],
                                            request_id='',
                                            trace_id=''))


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


def _launch(creation):
    return utils.launch(cluster_name_on_cloud='cloud',
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
                        use_spot=True,
                        creation=creation)


@nebius_sdk_required
def test_launch_notes_a_confirmed_create(instance_service):
    creation = {}
    assert _launch(creation) == 'instance-id'
    request = instance_service.create.call_args.args[0]
    assert creation == {
        'name': request.metadata.name,
        'accepted': True,
        'id': 'instance-id'
    }


@nebius_sdk_required
@pytest.mark.parametrize('code', [
    'INVALID_ARGUMENT', 'FAILED_PRECONDITION', 'PERMISSION_DENIED',
    'UNAUTHENTICATED', 'NOT_FOUND', 'RESOURCE_EXHAUSTED', 'OUT_OF_RANGE'
])
def test_launch_notes_a_refused_create(instance_service, code):
    instance_service.create.side_effect = _request_error(code)
    creation = {}
    with pytest.raises(Exception):
        _launch(creation)
    assert creation['refused'] is True
    assert 'accepted' not in creation


@nebius_sdk_required
@pytest.mark.parametrize('code', [
    'UNAVAILABLE', 'DEADLINE_EXCEEDED', 'UNKNOWN', 'INTERNAL', 'ABORTED',
    'CANCELLED', 'ALREADY_EXISTS'
])
def test_launch_create_with_unknown_outcome_is_not_refused(
        instance_service, code):
    instance_service.create.side_effect = _request_error(code)
    creation = {}
    with pytest.raises(Exception):
        _launch(creation)
    assert 'name' in creation
    assert not creation.get('refused')


@nebius_sdk_required
def test_launch_error_after_create_is_not_refused(instance_service):
    # The VM object exists: a failing status poll must not release it.
    instance_service.get_by_name.side_effect = _request_error('NOT_FOUND')
    creation = {}
    with pytest.raises(Exception):
        _launch(creation)
    assert creation['accepted'] is True
    assert not creation.get('refused')


@nebius_sdk_required
def test_launch_failure_before_create_notes_nothing(instance_service,
                                                    monkeypatch):

    def no_subnet(*_):
        raise ValueError('No subnets found')

    monkeypatch.setattr(utils, 'get_subnet_id', no_subnet)
    creation = {}
    with pytest.raises(ValueError):
        _launch(creation)
    assert creation == {}
    instance_service.create.assert_not_called()


# ------------------------------------------------------------------ config


@pytest.mark.parametrize('value', [1, 3, 100])
def test_schema_accepts_positive_integers(value):
    jsonschema.validate({'nebius': {
        'max_instance_creations': value
    }}, schemas.get_config_schema())


@pytest.mark.parametrize('value', [0, -1, 1.5, True, '3', None])
def test_schema_rejects_other_values(value):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'nebius': {
            'max_instance_creations': value
        }}, schemas.get_config_schema())


def test_task_override():
    task = sky.Task.from_yaml_config({
        'resources': {
            'infra': 'nebius',
        },
        'config': {
            'nebius': {
                'max_instance_creations': 2
            }
        }
    })
    assert next(
        iter(task.resources
            )).cluster_config_overrides['nebius']['max_instance_creations'] == 2
    assert (
        'nebius',
        'max_instance_creations') in constants.OVERRIDEABLE_CONFIG_KEYS_IN_TASK


def _render(**variables):
    path = pathlib.Path(sky.__file__).parent / 'templates' / 'nebius-ray.yml.j2'
    rendered = jinja2.Template(path.read_text()).render(
        num_nodes=1,
        credentials={},
        initial_setup_commands=[],
        docker_image=None,
        **variables)
    return rendered.split('auth:')[0]


@pytest.mark.parametrize('limit', [None, 2])
@mock.patch('sky.provision.nebius.utils.get_project_by_region',
            return_value='project-id')
def test_override_reaches_the_provider_config(mock_project, limit):
    del mock_project  # Keeps the deploy variables off the Nebius API.
    overrides = {} if limit is None else {
        'nebius': {
            'max_instance_creations': limit
        }
    }
    resources = resources_lib.Resources(
        cloud=nebius_cloud.Nebius(),
        instance_type='cpu-d3_4vcpu-16gb',
        region='eu-north1',
        _cluster_config_overrides=overrides).assert_launchable()
    region = mock.MagicMock()
    region.name = 'eu-north1'
    variables = nebius_cloud.Nebius().make_deploy_resources_variables(
        resources,
        resources_utils.ClusterName(display_name='c', name_on_cloud='c-1234'),
        region, None, 1)
    assert variables['max_instance_creations'] == limit
    provider = _render(**variables)
    if limit is None:
        assert 'max_instance_creations' not in provider
    else:
        assert f'\n  max_instance_creations: {limit}\n' in provider

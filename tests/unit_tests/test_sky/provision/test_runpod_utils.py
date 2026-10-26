"""Unit tests for sky.provision.runpod.utils."""
from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sky.provision.runpod import utils as runpod_utils


class TestCreateTemplateForDockerLogin:

    def test_no_docker_login_config_returns_image_unchanged(self):
        image, template_id = runpod_utils._create_template_for_docker_login(
            cluster_name='test-cluster',
            image_name='my-org/my-image:tag',
            docker_login_config=None,
        )
        assert image == 'my-org/my-image:tag'
        assert template_id is None

    def test_docker_login_config_passes_formatted_image_to_create_template(
            self):
        """Regression test for #9546.

        create_template must receive the fully-qualified image name, not None.
        Passing None caused Python to serialize it as the literal string "None"
        in the GraphQL mutation (imageName: "None"), which the RunPod API now
        rejects as an invalid image name.
        """
        mock_auth_resp = {'id': 'auth-id-123'}
        mock_template_resp = {'id': 'template-id-456'}

        with patch('sky.provision.runpod.utils.runpod') as mock_runpod:
            mock_runpod.runpod.create_container_registry_auth.return_value = (
                mock_auth_resp)
            mock_runpod.runpod.create_template.return_value = mock_template_resp

            image, template_id = (
                runpod_utils._create_template_for_docker_login(
                    cluster_name='test-cluster',
                    image_name='my-org/my-image:tag',
                    docker_login_config={
                        'username': 'user',
                        'password': 'pass',
                        'server': 'ghcr.io',
                    },
                ))

        assert image == 'ghcr.io/my-org/my-image:tag'
        assert template_id == 'template-id-456'

        # The critical assertion: create_template must not receive None or
        # the string "None" as image_name.
        mock_runpod.runpod.create_template.assert_called_once_with(
            name=mock.ANY,
            image_name='ghcr.io/my-org/my-image:tag',
            registry_auth_id='auth-id-123',
        )

    def test_image_already_has_server_prefix_not_doubled(self):
        mock_auth_resp = {'id': 'auth-id-123'}
        mock_template_resp = {'id': 'template-id-456'}

        with patch('sky.provision.runpod.utils.runpod') as mock_runpod:
            mock_runpod.runpod.create_container_registry_auth.return_value = (
                mock_auth_resp)
            mock_runpod.runpod.create_template.return_value = mock_template_resp

            image, _ = runpod_utils._create_template_for_docker_login(
                cluster_name='test-cluster',
                image_name='ghcr.io/my-org/my-image:tag',
                docker_login_config={
                    'username': 'user',
                    'password': 'pass',
                    'server': 'ghcr.io',
                },
            )

        # Server prefix should not be doubled.
        assert image == 'ghcr.io/my-org/my-image:tag'


class TestLaunchCudaVersions:
    """GraphQL host-version constraints remain opt-in and fail closed."""

    _COMMON_KWARGS = dict(cluster_name='test-cuda',
                          node_type='head',
                          region='US',
                          zone='US-TX-3',
                          disk_size=20,
                          image_name='runpod/pytorch:test',
                          ports=None,
                          public_key='ssh-ed25519 test',
                          preemptible=False,
                          bid_per_gpu=0.0,
                          docker_login_config=None)

    @pytest.mark.parametrize('versions', [None, ['12.8', '13.1']])
    def test_on_demand_gpu_versions(self, versions):
        with patch('sky.provision.runpod.utils.runpod') as mock_runpod:
            mock_runpod.runpod.get_gpu.return_value = {'memoryInGb': 8}
            mock_runpod.runpod.create_pod.return_value = {'id': 'test-pod'}
            result = runpod_utils.launch(instance_type='1x_RTX3070_COMMUNITY',
                                         allowed_cuda_versions=versions,
                                         **self._COMMON_KWARGS)
        assert result == 'test-pod'
        params = mock_runpod.runpod.create_pod.call_args.kwargs
        assert params['gpu_type_id'] == 'NVIDIA GeForce RTX 3070'
        assert params['gpu_count'] == 1
        assert params['cloud_type'] == 'COMMUNITY'
        assert params['support_public_ip'] is True
        if versions is None:
            assert 'allowed_cuda_versions' not in params
        else:
            assert params['allowed_cuda_versions'] == versions

    @pytest.mark.parametrize('instance_type,preemptible',
                             [('cpu5c-4-8', False),
                              ('1x_RTX3070_COMMUNITY', True)])
    def test_unsupported_paths_fail_before_provider_calls(
            self, instance_type, preemptible):
        kwargs = self._COMMON_KWARGS.copy()
        kwargs['preemptible'] = preemptible
        with patch('sky.provision.runpod.utils.runpod') as mock_runpod:
            with pytest.raises(ValueError, match='on-demand GPU'):
                runpod_utils.launch(instance_type=instance_type,
                                    allowed_cuda_versions=['12.8'],
                                    **kwargs)
            assert not mock_runpod.mock_calls

    def test_empty_allowlist_fails_before_provider_calls(self):
        with patch('sky.provision.runpod.utils.runpod') as mock_runpod:
            with pytest.raises(ValueError, match='must not be empty'):
                runpod_utils.launch(instance_type='1x_RTX3070_COMMUNITY',
                                    allowed_cuda_versions=[],
                                    **self._COMMON_KWARGS)
            assert not mock_runpod.mock_calls

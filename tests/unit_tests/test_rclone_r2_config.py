"""The R2 rclone profile must work with a bucket-scoped (object-level) token."""
import collections
from unittest import mock

from sky.data import data_utils


def test_r2_rclone_profile_skips_bucket_check_and_creation():
    # Cloudflare: tokens with object-level permissions need
    # `no_check_bucket = true`, or rclone fails trying to create the bucket.
    # SkyPilot has already verified the bucket exists before mounting.
    creds = collections.namedtuple('Creds', 'access_key secret_key')('a', 's')
    with mock.patch.object(data_utils.cloudflare, 'session'), \
            mock.patch.object(data_utils.cloudflare, 'get_r2_credentials',
                              return_value=creds), \
            mock.patch.object(data_utils.cloudflare, 'create_endpoint',
                              return_value='https://acct.r2.example'):
        config = data_utils.Rclone.RcloneStores.R2.get_config(
            rclone_profile_name='sky-r2-b')
    lines = [line.strip() for line in config.splitlines()]
    assert lines[0] == '[sky-r2-b]'
    assert 'provider = Cloudflare' in lines
    assert 'no_check_bucket = true' in lines

"""Vast cloud adaptor."""

import functools

_vast_sdk = None


class _VastClient:
    """Preserve malformed inventory responses instead of treating them as empty."""

    def __init__(self, sdk):
        self._sdk = sdk

    def __getattr__(self, name):
        return getattr(self._sdk, name)

    def show_instances(self):
        rows = []
        params = {'limit': 25, 'order_by': [{'col': 'id', 'dir': 'asc'}]}
        tokens = set()
        while True:
            page = self._sdk.show_instances_v1(params)
            if (not isinstance(page, dict) or
                    not isinstance(page.get('instances'), list)):
                raise RuntimeError('Ambiguous Vast inventory page')
            rows.extend(page['instances'])
            token = page.get('next_token')
            if token is None:
                return rows
            if not isinstance(token, str) or not token or token in tokens:
                raise RuntimeError('Ambiguous Vast inventory pagination')
            tokens.add(token)
            params = dict(params, after_token=token)


def import_package(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _vast_sdk

        if _vast_sdk is None:
            try:
                import vastai as _vast  # pylint: disable=import-outside-toplevel
                _vast_sdk = _VastClient(_vast.VastAI())
            except ImportError as e:
                raise ImportError(f'Fail to import dependencies for vast: {e}\n'
                                  'Try pip install "skypilot[vast]"') from None
        return func(*args, **kwargs)

    return wrapper


@import_package
def vast():
    """Return the vast package."""
    return _vast_sdk

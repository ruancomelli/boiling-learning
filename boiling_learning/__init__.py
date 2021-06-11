from importlib_metadata import version as _version
from semver import VersionInfo

__all__ = ['__version__', 'version_info', 'version_compact']

__version__ = _version('boiling_learning')
version_info = VersionInfo.parse(__version__)
version_compact = __version__.replace('-', '')

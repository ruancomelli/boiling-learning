from importlib_metadata import version as _version
from semver import VersionInfo

# pre-import common subpackages
import boiling_learning.datasets
import boiling_learning.io
import boiling_learning.management
import boiling_learning.models
import boiling_learning.preprocessing
import boiling_learning.utils

__all__ = ['__version__', 'version_info', 'version_compact']

__version__: str = _version('boiling_learning')
version_info = VersionInfo.parse(__version__)
version_compact = __version__.replace('-', '')

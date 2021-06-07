from pathlib import Path

from semver import VersionInfo

_version_file = Path(__file__).resolve().parents[1] / 'VERSION'

__version__ = _version_file.read_text().strip()
version_info = VersionInfo.parse(__version__)
__version_compact__ = ''.join(__version__.split('-'))

"""Version info for fe2pv.
On the ``main`` branch, use 'dev0' to denote a development version.
For example:
version_info = 0, 1, 'dev0'
---
When generating pre-release wheels, use '0rcN', for example:
version_info = 0, 2, '0rc1'
Denotes the first release candidate.
"""
# major, minor, patch
version_info = 0, 0, "dev0"

# Nice string for the version
__version__ = ".".join(map(str, version_info))

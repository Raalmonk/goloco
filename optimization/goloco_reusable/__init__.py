"""GOLOCO inference from compressed CERES features to the original table."""
from .backend import HeadlessGoloco, PreparedInput, SourceIdentityError

__all__ = ['HeadlessGoloco', 'PreparedInput', 'SourceIdentityError']
__version__ = '0.2.0'


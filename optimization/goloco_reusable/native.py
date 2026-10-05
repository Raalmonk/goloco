"""Guarded Python and C0 Rust forest adapters; no benchmark dependencies."""
import ctypes
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted


class NativeUnavailableError(RuntimeError):
    pass


class FastForest:
    """Original validation and ordered, unchanged sklearn Cython tree calls."""
    def __init__(self, estimator):
        if type(estimator) is not RandomForestRegressor:
            raise TypeError('python_fast/rust require exact RandomForestRegressor models')
        check_is_fitted(estimator)
        if estimator.n_outputs_ != 1:
            raise ValueError('Only the released single-output regressors are supported')
        if joblib.effective_n_jobs(estimator.n_jobs) != 1:
            raise ValueError('The original estimator must have one effective worker')
        if not estimator.estimators_:
            raise ValueError('The forest must contain fitted trees')
        for tree in estimator.estimators_:
            if type(tree) is not DecisionTreeRegressor:
                raise TypeError('Forest members must be exact DecisionTreeRegressor instances')
            check_is_fitted(tree)
            if (tree.n_outputs_ != 1 or tree.n_features_ != estimator.n_features_
                    or tree.tree_.n_outputs != 1 or tree.tree_.n_features != estimator.n_features_
                    or tree.tree_.value.shape[1:] != (1, 1)):
                raise ValueError('Tree dimensions differ from the guarded forest')
        self.estimator = estimator

    def predict(self, matrix):
        checked = self.estimator._validate_X_predict(matrix)
        result = np.zeros(checked.shape[0], dtype=np.float64)
        for tree in self.estimator.estimators_:
            result += tree.tree_.predict(checked)[:, 0]
        result /= len(self.estimator.estimators_)
        return result


def export_size(estimator):
    return sum(tree.tree_.node_count for tree in estimator.estimators_) * 5 * 8 + len(estimator.estimators_) * 8


def pack_forest(estimator):
    """Export the unchanged C0 layout with model-local nodes and roots."""
    FastForest(estimator)
    sizes = [tree.tree_.node_count for tree in estimator.estimators_]
    count = sum(sizes)
    arrays = {name: np.empty(count, dtype=np.int64 if name in ['left', 'right', 'feature'] else np.float64)
              for name in ['left', 'right', 'feature', 'threshold', 'value']}
    arrays['roots'] = np.empty(len(sizes), dtype=np.int64)
    offset = 0
    for index, tree in enumerate(estimator.estimators_):
        native = tree.tree_
        n = native.node_count
        part = slice(offset, offset + n)
        left, right = native.children_left, native.children_right
        if not np.array_equal(left == -1, right == -1):
            raise ValueError('Inconsistent leaf children')
        internal = left != -1
        if not (np.all((left[internal] >= 0) & (left[internal] < n))
                and np.all((right[internal] >= 0) & (right[internal] < n))
                and np.all((native.feature[internal] >= 0) & (native.feature[internal] < estimator.n_features_))):
            raise ValueError('Invalid source tree topology')
        arrays['roots'][index] = offset
        arrays['left'][part] = np.where(left == -1, -1, left + offset)
        arrays['right'][part] = np.where(right == -1, -1, right + offset)
        arrays['feature'][part] = native.feature
        arrays['threshold'][part] = native.threshold
        arrays['value'][part] = native.value[:, 0, 0]
        offset += n
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


class RustLibrary:
    """Explicitly load the requested C0 ABI; never fall back to another backend."""
    def __init__(self, path):
        if path is None:
            raise NativeUnavailableError('backend="rust" requires an explicit compiled library path')
        self.path = Path(path).resolve()
        try:
            self.handle = ctypes.CDLL(str(self.path))
            self.handle.goloco_forest_abi_version.restype = ctypes.c_uint32
            if self.handle.goloco_forest_abi_version() != 1:
                raise NativeUnavailableError('The requested library has an incompatible GOLOCO ABI')
            self.predict = self.handle.goloco_predict_forest
            self.predict.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_size_t] * 2
                                     + [ctypes.c_void_p] + [ctypes.c_size_t] * 2
                                     + [ctypes.c_int64] * 2 + [ctypes.c_void_p])
            self.predict.restype = ctypes.c_int32
        except (OSError, AttributeError) as exc:
            raise NativeUnavailableError('Cannot use requested GOLOCO Rust library %s: %s' % (self.path, exc)) from exc


class RustForest:
    """C0 prediction with original estimator validation and owned read-only arrays."""
    def __init__(self, estimator, library, arrays=None):
        FastForest(estimator)
        self.estimator, self.library = estimator, library
        self.n_nodes = sum(tree.tree_.node_count for tree in estimator.estimators_)
        self.n_trees = len(estimator.estimators_)
        self.owns_export = arrays is None
        self.arrays = pack_forest(estimator) if arrays is None else dict(arrays)
        for name in ['left', 'right', 'feature', 'threshold', 'value', 'roots']:
            array = self.arrays[name]
            expected_dtype = np.dtype(np.int64 if name in ['left', 'right', 'feature', 'roots'] else np.float64)
            expected_length = self.n_trees if name == 'roots' else self.n_nodes
            if (not isinstance(array, np.ndarray) or array.ndim != 1
                    or array.dtype != expected_dtype or len(array) != expected_length
                    or not array.flags.c_contiguous or not array.flags.aligned
                    or not array.dtype.isnative or array.flags.writeable):
                raise ValueError('Invalid read-only native parameter array: ' + name)
        self.packed_bytes = sum(array.nbytes for array in self.arrays.values())
        if self.packed_bytes != export_size(estimator):
            raise ValueError('Native parameter size differs from source forest')
        self.pointers = [self.arrays[name].ctypes.data for name in ['left', 'right', 'feature', 'threshold', 'value', 'roots']]
        self.last_call_metrics = {}

    def predict(self, matrix):
        checked = self.estimator._validate_X_predict(matrix)
        if not isinstance(checked, np.ndarray):
            raise TypeError('RustForest supports the dense GOLOCO matrix path')
        if checked.dtype != np.float32 or checked.ndim != 2:
            raise ValueError('Legacy validation did not produce a two-dimensional float32 matrix')
        output = np.empty(checked.shape[0], dtype=np.float64)
        status = self.library.predict(*self.pointers, self.n_nodes, self.n_trees,
                                      checked.ctypes.data, checked.shape[0], checked.shape[1],
                                      checked.strides[0], checked.strides[1], output.ctypes.data)
        if status:
            raise ValueError('Rust forest structural/argument error ' + str(status))
        copied = 0 if np.shares_memory(checked, matrix) else int(checked.nbytes)
        self.last_call_metrics = {'ffi_calls': 1, 'input_conversion_bytes': copied,
                                  'output_allocation_bytes': int(output.nbytes),
                                  'tracked_numpy_allocations': 1 + int(bool(copied))}
        return output

//! Ordered evaluation of already validated, unchanged single-output forests.
//! The caller owns all arrays throughout this call. No pickle parsing occurs.

#[no_mangle]
pub extern "C" fn goloco_forest_abi_version() -> u32 { 1 }

/// Return 0 on success, otherwise a structural/argument error. No allocation,
/// threads, retained pointers, floating-point reassociation, or fast math.
#[no_mangle]
pub unsafe extern "C" fn goloco_predict_forest(
    left: *const i64,
    right: *const i64,
    feature: *const i64,
    threshold: *const f64,
    value: *const f64,
    roots: *const i64,
    n_nodes: usize,
    n_trees: usize,
    input: *const u8,
    n_samples: usize,
    n_features: usize,
    sample_stride_bytes: i64,
    feature_stride_bytes: i64,
    output: *mut f64,
) -> i32 {
    if left.is_null() || right.is_null() || feature.is_null() || threshold.is_null()
        || value.is_null() || roots.is_null() || input.is_null() || output.is_null()
        || n_nodes == 0 || n_trees == 0 || n_samples == 0 || n_features == 0 {
        return 1;
    }
    for sample in 0..n_samples {
        let mut sum = 0.0_f64;
        // Each sample accumulates its leaves in the original estimator order.
        for tree in 0..n_trees {
            let mut node = roots.add(tree).read();
            let mut steps = 0_usize;
            loop {
                if node < 0 || node as usize >= n_nodes { return 2; }
                let index = node as usize;
                let child = left.add(index).read();
                if child == -1 {
                    sum += value.add(index).read();
                    break;
                }
                let column = feature.add(index).read();
                if column < 0 || column as usize >= n_features { return 3; }
                let row_offset = match (sample as i64).checked_mul(sample_stride_bytes) {
                    Some(x) => x, None => return 4,
                };
                let column_offset = match column.checked_mul(feature_stride_bytes) {
                    Some(x) => x, None => return 4,
                };
                let byte_offset = match row_offset.checked_add(column_offset) {
                    Some(x) => x, None => return 4,
                };
                // NumPy may preserve negative strides or unaligned buffers.
                let x = (input.offset(byte_offset as isize) as *const f32).read_unaligned();
                node = if (x as f64) <= threshold.add(index).read() {
                    child
                } else {
                    right.add(index).read()
                };
                steps += 1;
                if steps > n_nodes { return 5; }
            }
        }
        output.add(sample).write(sum / n_trees as f64);
    }
    0
}

/// Diagnostic only: C0 remains unchanged; duration includes native checks,
/// traversal and ordered accumulation. This entry is never used in headline runs.
#[no_mangle]
pub unsafe extern "C" fn goloco_predict_forest_profile(
    left: *const i64, right: *const i64, feature: *const i64,
    threshold: *const f64, value: *const f64, roots: *const i64,
    n_nodes: usize, n_trees: usize, input: *const u8,
    n_samples: usize, n_features: usize, sample_stride_bytes: i64,
    feature_stride_bytes: i64, output: *mut f64, seconds: *mut f64,
) -> i32 {
    if seconds.is_null() { return 20; }
    let start = std::time::Instant::now();
    let code = goloco_predict_forest(left, right, feature, threshold, value,
        roots, n_nodes, n_trees, input, n_samples, n_features,
        sample_stride_bytes, feature_stride_bytes, output);
    seconds.write(start.elapsed().as_secs_f64());
    code
}

/// Each descriptor is owned by the Python wrapper for the complete batch call.
/// Explicit lengths and byte bounds are checked before using the C0 kernel.
#[repr(C)]
pub struct ForestRequest {
    pub left: *const i64,
    pub right: *const i64,
    pub feature: *const i64,
    pub threshold: *const f64,
    pub value: *const f64,
    pub roots: *const i64,
    pub left_len: usize,
    pub right_len: usize,
    pub feature_len: usize,
    pub threshold_len: usize,
    pub value_len: usize,
    pub roots_len: usize,
    pub n_nodes: usize,
    pub n_trees: usize,
    pub input: *const u8,
    pub input_low: usize,
    pub input_high: usize,
    pub n_samples: usize,
    pub n_features: usize,
    pub sample_stride_bytes: i64,
    pub feature_stride_bytes: i64,
    pub output: *mut f64,
    pub output_len: usize,
}

#[no_mangle]
pub extern "C" fn goloco_batch_descriptor_size() -> usize {
    std::mem::size_of::<ForestRequest>()
}

#[no_mangle]
pub unsafe extern "C" fn goloco_predict_batch(
    requests: *const ForestRequest, count: usize, failed_index: *mut usize,
) -> i32 {
    if requests.is_null() || failed_index.is_null() || count == 0
       || count > 4096 { return 30; }
    let descriptor_bytes = match count.checked_mul(std::mem::size_of::<ForestRequest>()) {
        Some(v) if v <= isize::MAX as usize => v, _ => return 34,
    };
    if (requests as usize).checked_add(descriptor_bytes).is_none()
        || (failed_index as usize).checked_add(std::mem::size_of::<usize>()).is_none() {
        return 34;
    }
    failed_index.write(usize::MAX);
    for index in 0..count {
        failed_index.write(index);
        let r = &*requests.add(index);
        if r.left_len != r.n_nodes || r.right_len != r.n_nodes
            || r.feature_len != r.n_nodes || r.threshold_len != r.n_nodes
            || r.value_len != r.n_nodes || r.roots_len != r.n_trees
            || r.output_len < r.n_samples || r.n_samples == 0 || r.n_features == 0 {
            return 31;
        }
        // Every .add()/read()/write() span below must fit Rust's isize bound
        // and must not wrap its pointer address. This check precedes all
        // parameter-array, matrix or output dereferences.
        let spans = [
            (r.left as usize, r.left_len), (r.right as usize, r.right_len),
            (r.feature as usize, r.feature_len), (r.threshold as usize, r.threshold_len),
            (r.value as usize, r.value_len), (r.roots as usize, r.roots_len),
            (r.output as usize, r.output_len),
        ];
        for (pointer, length) in spans {
            if length > isize::MAX as usize / 8 { return 34; }
            let bytes = match length.checked_mul(8) { Some(v) => v, None => return 34 };
            if pointer.checked_add(bytes).is_none() { return 34; }
        }
        let ns = match i64::try_from(r.n_samples - 1) { Ok(v) => v, Err(_) => return 32 };
        let nf = match i64::try_from(r.n_features - 1) { Ok(v) => v, Err(_) => return 32 };
        let rows = match ns.checked_mul(r.sample_stride_bytes) { Some(v) => v, None => return 32 };
        let cols = match nf.checked_mul(r.feature_stride_bytes) { Some(v) => v, None => return 32 };
        let lower = match rows.min(0).checked_add(cols.min(0)) { Some(v) => v, None => return 32 };
        let upper = match rows.max(0).checked_add(cols.max(0)).and_then(|v| v.checked_add(4)) {
            Some(v) => v, None => return 32,
        };
        let base = r.input as usize as i128;
        let low = base + lower as i128;
        let high = base + upper as i128;
        if r.input.is_null() || low < r.input_low as i128 || high > r.input_high as i128
            || low < 0 || high < low || r.input_high < r.input_low {
            return 33;
        }
        let status = goloco_predict_forest(r.left, r.right, r.feature, r.threshold,
            r.value, r.roots, r.n_nodes, r.n_trees, r.input, r.n_samples,
            r.n_features, r.sample_stride_bytes, r.feature_stride_bytes, r.output);
        if status != 0 { return status; }
    }
    failed_index.write(usize::MAX);
    0
}

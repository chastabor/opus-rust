use super::{Conv2dLayer, LinearLayer, WeightArray, WeightType, WEIGHT_BLOCK_SIZE};

const SPARSE_BLOCK_SIZE: usize = 32;

/// Error returned when weight initialization fails (missing or wrong-sized array).
#[derive(Debug, Clone)]
pub struct WeightError;

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "weight initialization failed: missing or wrong-sized array")
    }
}

/// Binary weight blob header, matches C `WeightHead`.
struct WeightHead {
    #[allow(dead_code)]
    head: [u8; 4],
    #[allow(dead_code)]
    version: i32,
    weight_type: i32,
    size: i32,
    block_size: i32,
    name: [u8; 44],
}

impl WeightHead {
    fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < WEIGHT_BLOCK_SIZE as usize {
            return None;
        }
        let head = [data[0], data[1], data[2], data[3]];
        let version = i32::from_le_bytes(data[4..8].try_into().ok()?);
        let weight_type = i32::from_le_bytes(data[8..12].try_into().ok()?);
        let size = i32::from_le_bytes(data[12..16].try_into().ok()?);
        let block_size = i32::from_le_bytes(data[16..20].try_into().ok()?);
        let mut name = [0u8; 44];
        name.copy_from_slice(&data[20..64]);
        Some(WeightHead { head, version, weight_type, size, block_size, name })
    }

    fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(self.name.len());
        std::str::from_utf8(&self.name[..end]).unwrap_or("")
    }

    fn to_weight_type(&self) -> Option<WeightType> {
        match self.weight_type {
            0 => Some(WeightType::Float),
            1 => Some(WeightType::Int),
            2 => Some(WeightType::QWeight),
            3 => Some(WeightType::Int8),
            _ => None,
        }
    }
}

/// Parse a single weight record from a binary blob.
/// Advances `offset` past the consumed record.
/// Matches C `parse_record` from parse_lpcnet_weights.c.
fn parse_record(data: &[u8], offset: &mut usize) -> Option<WeightArray> {
    let block_size = WEIGHT_BLOCK_SIZE as usize;
    let remaining = data.len() - *offset;
    if remaining < block_size {
        return None;
    }

    let h = WeightHead::from_bytes(&data[*offset..])?;

    if h.block_size < h.size {
        return None;
    }
    if h.block_size as usize > remaining - block_size {
        return None;
    }
    if h.size < 0 {
        return None;
    }
    if h.name[43] != 0 {
        return None;
    }

    let name = h.name_str().to_string();
    let weight_type = h.to_weight_type()?;
    let data_start = *offset + block_size;
    let data_end = data_start + h.size as usize;
    let array_data = data[data_start..data_end].to_vec();

    *offset += block_size + h.block_size as usize;

    Some(WeightArray {
        name,
        weight_type,
        data: array_data,
    })
}

/// Parse all weight arrays from a binary weight blob.
/// Matches C `parse_weights` from parse_lpcnet_weights.c.
/// Returns None if the blob is malformed.
pub fn parse_weights(data: &[u8]) -> Option<Vec<WeightArray>> {
    let mut arrays = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let array = parse_record(data, &mut offset)?;
        arrays.push(array);
    }
    Some(arrays)
}

fn find_array_entry<'a>(arrays: &'a [WeightArray], name: &str) -> Option<&'a WeightArray> {
    arrays.iter().find(|a| a.name == name)
}

fn find_array_check<'a>(arrays: &'a [WeightArray], name: &str, expected_size: usize) -> Option<&'a [u8]> {
    let a = find_array_entry(arrays, name)?;
    if a.data.len() == expected_size {
        Some(&a.data)
    } else {
        None
    }
}

/// Find an optional weight array. Returns Ok(None) if not found,
/// Err if found with wrong size.
fn opt_array_check<'a>(arrays: &'a [WeightArray], name: &str, expected_size: usize) -> Result<Option<&'a [u8]>, WeightError> {
    match find_array_entry(arrays, name) {
        Some(a) => {
            if a.data.len() == expected_size {
                Ok(Some(&a.data))
            } else {
                Err(WeightError)
            }
        }
        None => Ok(None),
    }
}

/// Validate a sparse index array and count total blocks.
/// Matches C `find_idx_check` from parse_lpcnet_weights.c.
fn find_idx_check(
    arrays: &[WeightArray],
    name: &str,
    nb_in: usize,
    mut nb_out: usize,
) -> Option<(Vec<i32>, usize)> {
    let a = find_array_entry(arrays, name)?;
    let idx = bytes_to_i32_vec(&a.data);
    let mut total_blocks = 0usize;
    let mut pos = 0;
    let mut remain = idx.len();

    while remain > 0 {
        let nb_blocks = idx[pos] as usize;
        pos += 1;
        if remain < nb_blocks + 1 {
            return None;
        }
        for _ in 0..nb_blocks {
            let p = idx[pos] as usize;
            pos += 1;
            if p + 3 >= nb_in || (p & 0x3) != 0 {
                return None;
            }
        }
        if nb_out < 8 {
            return None;
        }
        nb_out -= 8;
        remain -= nb_blocks + 1;
        total_blocks += nb_blocks;
    }
    if nb_out != 0 {
        return None;
    }
    Some((idx, total_blocks))
}

fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bytes_to_i32_vec(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bytes_to_i8_vec(data: &[u8]) -> Vec<i8> {
    data.iter().map(|&b| b as i8).collect()
}

/// Infer output dimension from a float bias array's byte size.
/// Requires the array to be `WeightType::Float`.
pub fn weight_output_dim(arrays: &[WeightArray], name: &str) -> Result<usize, WeightError> {
    let a = arrays.iter().find(|a| a.name == name).ok_or(WeightError)?;
    if a.weight_type != WeightType::Float { return Err(WeightError); }
    Ok(a.data.len() / 4)
}

/// Infer the input dimension of a float dense layer from its weight matrix size.
/// weight_matrix_bytes = nb_inputs * nb_outputs * 4, so nb_inputs = bytes / (nb_outputs * 4).
pub fn weight_input_dim(arrays: &[WeightArray], weights_name: &str, nb_outputs: usize) -> Result<usize, WeightError> {
    let a = arrays.iter().find(|a| a.name == weights_name).ok_or(WeightError)?;
    if nb_outputs == 0 { return Err(WeightError); }
    Ok(a.data.len() / (nb_outputs * 4))
}

/// Initialize a LinearLayer from a WeightArray list.
/// Matches C `linear_init` from parse_lpcnet_weights.c.
pub fn linear_init(
    arrays: &[WeightArray],
    bias: Option<&str>,
    weights: Option<&str>,
    float_weights: Option<&str>,
    weights_idx: Option<&str>,
    diag: Option<&str>,
    scale: Option<&str>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer, WeightError> {
    let mut layer = LinearLayer::new(nb_inputs, nb_outputs);

    if let Some(name) = bias {
        let data = find_array_check(arrays, name, nb_outputs * 4).ok_or(WeightError)?;
        layer.bias = Some(bytes_to_f32_vec(data));
    }

    if let Some(idx_name) = weights_idx {
        let (idx_data, total_blocks) = find_idx_check(arrays, idx_name, nb_inputs, nb_outputs).ok_or(WeightError)?;
        layer.weights_idx = Some(idx_data);

        if let Some(name) = weights {
            let data = find_array_check(arrays, name, SPARSE_BLOCK_SIZE * total_blocks).ok_or(WeightError)?;
            layer.weights = Some(bytes_to_i8_vec(data));
        }
        if let Some(name) = float_weights
            && let Some(data) = opt_array_check(arrays, name, SPARSE_BLOCK_SIZE * total_blocks * 4).map_err(|_| WeightError)?
        {
            layer.float_weights = Some(bytes_to_f32_vec(data));
        }
    } else {
        if let Some(name) = weights {
            let data = find_array_check(arrays, name, nb_inputs * nb_outputs).ok_or(WeightError)?;
            layer.weights = Some(bytes_to_i8_vec(data));
        }
        if let Some(name) = float_weights
            && let Some(data) = opt_array_check(arrays, name, nb_inputs * nb_outputs * 4).map_err(|_| WeightError)?
        {
            layer.float_weights = Some(bytes_to_f32_vec(data));
        }
    }

    if let Some(name) = diag {
        let data = find_array_check(arrays, name, nb_outputs * 4).ok_or(WeightError)?;
        layer.diag = Some(bytes_to_f32_vec(data));
    }

    if weights.is_some() {
        let name = scale.ok_or(WeightError)?;
        let data = find_array_check(arrays, name, nb_outputs * 4).ok_or(WeightError)?;
        layer.scale = Some(bytes_to_f32_vec(data));
    }

    Ok(layer)
}

/// Initialize a Conv2dLayer from a WeightArray list.
/// Matches C `conv2d_init` from parse_lpcnet_weights.c.
pub fn conv2d_init(
    arrays: &[WeightArray],
    bias: Option<&str>,
    float_weights: Option<&str>,
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
) -> Result<Conv2dLayer, WeightError> {
    let mut layer = Conv2dLayer::new(in_channels, out_channels, ktime, kheight);

    if let Some(name) = bias {
        let data = find_array_check(arrays, name, out_channels * 4).ok_or(WeightError)?;
        layer.bias = Some(bytes_to_f32_vec(data));
    }
    if let Some(name) = float_weights {
        let expected = in_channels * out_channels * ktime * kheight * 4;
        if let Some(data) = opt_array_check(arrays, name, expected).map_err(|_| WeightError)? {
            layer.float_weights = Some(bytes_to_f32_vec(data));
        }
    }

    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = WEIGHT_BLOCK_SIZE as usize;

    fn make_weight_record(name: &str, data: &[u8]) -> Vec<u8> {
        let mut record = vec![0u8; BLOCK];
        record[0..4].copy_from_slice(b"wght");
        record[4..8].copy_from_slice(&0i32.to_le_bytes());
        record[8..12].copy_from_slice(&0i32.to_le_bytes());
        let size = data.len() as i32;
        record[12..16].copy_from_slice(&size.to_le_bytes());
        let block_size = data.len().div_ceil(BLOCK) * BLOCK;
        record[16..20].copy_from_slice(&(block_size as i32).to_le_bytes());
        let name_bytes = name.as_bytes();
        let copy_len = name_bytes.len().min(43);
        record[20..20 + copy_len].copy_from_slice(&name_bytes[..copy_len]);
        record.extend_from_slice(data);
        record.resize(BLOCK + block_size, 0);
        record
    }

    #[test]
    fn test_parse_weights_single_record() {
        let data = 42.0f32.to_le_bytes();
        let blob = make_weight_record("test_bias", &data);
        let arrays = parse_weights(&blob).unwrap();
        assert_eq!(arrays.len(), 1);
        assert_eq!(arrays[0].name, "test_bias");
        assert_eq!(arrays[0].data.len(), 4);
        let val = f32::from_le_bytes(arrays[0].data[..4].try_into().unwrap());
        assert_eq!(val, 42.0);
    }

    #[test]
    fn test_parse_weights_multiple_records() {
        let mut blob = make_weight_record("bias", &1.0f32.to_le_bytes());
        blob.extend(make_weight_record("weights", &[0u8; 32]));
        let arrays = parse_weights(&blob).unwrap();
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].name, "bias");
        assert_eq!(arrays[1].name, "weights");
    }

    #[test]
    fn test_parse_weights_empty_blob() {
        let arrays = parse_weights(&[]).unwrap();
        assert!(arrays.is_empty());
    }

    #[test]
    fn test_linear_init_float_dense() {
        let bias_data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let weights_data: Vec<u8> = (0..32).flat_map(|i| (i as f32).to_le_bytes()).collect();

        let mut blob = make_weight_record("layer1_bias", &bias_data);
        blob.extend(make_weight_record("layer1_weights", &weights_data));
        let arrays = parse_weights(&blob).unwrap();

        let layer = linear_init(
            &arrays,
            Some("layer1_bias"),
            None,
            Some("layer1_weights"),
            None,
            None,
            None,
            4,
            8,
        )
        .unwrap();

        assert_eq!(layer.nb_inputs, 4);
        assert_eq!(layer.nb_outputs, 8);
        assert!(layer.bias.is_some());
        assert!(layer.float_weights.is_some());
        assert_eq!(layer.bias.as_ref().unwrap().len(), 8);
        assert_eq!(layer.float_weights.as_ref().unwrap().len(), 32);
    }

    #[test]
    fn test_linear_init_missing_array_fails() {
        let arrays = parse_weights(&[]).unwrap();
        let result = linear_init(
            &arrays,
            Some("nonexistent_bias"),
            None, None, None, None, None,
            4, 8,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_init() {
        let bias_data: Vec<u8> = (0..3).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let weights_data: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();

        let mut blob = make_weight_record("conv_bias", &bias_data);
        blob.extend(make_weight_record("conv_weights", &weights_data));
        let arrays = parse_weights(&blob).unwrap();

        let layer = conv2d_init(
            &arrays,
            Some("conv_bias"),
            Some("conv_weights"),
            2, 3, 1, 1,
        )
        .unwrap();

        assert_eq!(layer.in_channels, 2);
        assert_eq!(layer.out_channels, 3);
        assert!(layer.bias.is_some());
        assert!(layer.float_weights.is_some());
    }
}

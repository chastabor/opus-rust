// Parse C weight data files and emit binary blobs loadable by Rust `parse_weights`.
// Pure-Rust tool — no C compiler needed.

const WEIGHT_BLOCK_SIZE: usize = 64;
const WEIGHT_TYPE_FLOAT: i32 = 0;
const WEIGHT_TYPE_INT: i32 = 1;
const WEIGHT_TYPE_QWEIGHT: i32 = 2;
const WEIGHT_TYPE_INT8: i32 = 3;

/// Extract all static const array definitions from C source.
/// Returns map of array_name -> (element_type, raw_bytes).
fn extract_arrays(source: &str) -> HashMap<String, (i32, Vec<u8>)> {
    let mut arrays = HashMap::new();
    let mut pos = 0;

    while pos < source.len() {
        // Look for "static const <type> <name>[" pattern
        if let Some(start) = source[pos..].find("static const ") {
            let line_start = pos + start;
            let after_const = &source[line_start + 13..];

            // Extract type and name
            let (elem_type, type_len) = if after_const.starts_with("opus_int8 ") || after_const.starts_with("opus_int8\t") {
                (WEIGHT_TYPE_INT8, 10)
            } else if after_const.starts_with("float ") || after_const.starts_with("float\t") {
                (WEIGHT_TYPE_FLOAT, 6)
            } else if after_const.starts_with("opus_int16 ") || after_const.starts_with("opus_int16\t") {
                (WEIGHT_TYPE_INT, 11)
            } else if after_const.starts_with("int ") || after_const.starts_with("int\t") {
                (WEIGHT_TYPE_INT, 4)
            } else if after_const.starts_with("opus_uint8 ") || after_const.starts_with("opus_uint8\t") {
                (WEIGHT_TYPE_INT8, 11)
            } else {
                pos = line_start + 13;
                continue;
            };

            let rest = &after_const[type_len..];
            // Extract array name (up to '[')
            if let Some(bracket) = rest.find('[') {
                let name = rest[..bracket].trim().to_string();

                // Find the opening '{' and closing '};'
                let search_start = line_start + 13 + type_len + bracket;
                if let Some(brace_start) = source[search_start..].find('{') {
                    let data_start = search_start + brace_start + 1;
                    if let Some(brace_end) = source[data_start..].find("};") {
                        let values_str = &source[data_start..data_start + brace_end];

                        let bytes = parse_array_values(values_str, elem_type);
                        arrays.insert(name, (elem_type, bytes));

                        pos = data_start + brace_end + 2;
                        continue;
                    }
                }
            }
            pos = line_start + 13;
        } else {
            break;
        }
    }

    arrays
}

/// Parse comma-separated C integer/float literals into raw bytes.
fn parse_array_values(values_str: &str, elem_type: i32) -> Vec<u8> {
    let mut bytes = Vec::new();

    for token in values_str.split(',') {
        let s = token.trim();
        if s.is_empty() || s.starts_with("/*") || s.starts_with("//") {
            continue;
        }
        // Strip inline comments
        let s = if let Some(idx) = s.find("/*") { &s[..idx] } else { s };
        let s = s.trim();
        if s.is_empty() { continue; }

        match elem_type {
            WEIGHT_TYPE_INT8 => {
                // opus_int8 or opus_uint8: parse as i32 then truncate to i8/u8
                if let Ok(v) = s.parse::<i32>() {
                    bytes.push(v as u8);
                }
            }
            WEIGHT_TYPE_FLOAT => {
                // float: parse and write as little-endian f32
                let s = s.trim_end_matches('f').trim_end_matches('F');
                if let Ok(v) = s.parse::<f32>() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            WEIGHT_TYPE_INT | WEIGHT_TYPE_QWEIGHT => {
                // int: parse as i32, write as little-endian
                if let Ok(v) = s.parse::<i32>() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            _ => {}
        }
    }

    bytes
}

/// Extract WeightArray table entries from C source.
/// Returns list of (name, type, size, data_array_name).
///
/// Handles C preprocessor guards (`#ifdef`/`#endif`) around each entry and
/// resolves `WEIGHTS_*_TYPE` macro references via `#define` lookup.
fn extract_weight_table(source: &str, table_name: &str) -> Vec<(String, i32, usize, String)> {
    let mut entries = Vec::new();

    // Pre-parse #define WEIGHTS_*_TYPE macros so we can resolve type references.
    // Example: #define WEIGHTS_cond_net_pembed_bias_TYPE WEIGHT_TYPE_float
    let mut type_macros: HashMap<String, String> = HashMap::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#define WEIGHTS_")
            && let Some(type_pos) = rest.find("_TYPE ")
        {
            let macro_name = format!("WEIGHTS_{}", &rest[..type_pos + 5]);
            let type_val = rest[type_pos + 6..].trim().to_string();
            type_macros.insert(macro_name, type_val);
        }
    }

    // Find "const WeightArray <table_name>[] = {"
    let pattern = format!("WeightArray {}[]", table_name);
    let Some(table_start) = source.find(&pattern) else { return entries };
    let Some(brace) = source[table_start..].find('{') else { return entries };
    let data_start = table_start + brace + 1;
    let Some(end) = source[data_start..].find("};") else { return entries };
    let table_str = &source[data_start..data_start + end];

    // Strip preprocessor directive lines (#ifdef, #endif, #ifndef).
    let cleaned: String = table_str.lines()
        .filter(|l| !l.trim().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");

    // Find each {field1, field2, field3, field4} entry via balanced braces.
    let mut pos = 0;
    while pos < cleaned.len() {
        let Some(open) = cleaned[pos..].find('{') else { break };
        let entry_start = pos + open + 1;
        let Some(close) = cleaned[entry_start..].find('}') else { break };
        let entry = cleaned[entry_start..entry_start + close].trim();
        pos = entry_start + close + 1;

        if entry.is_empty() || entry.starts_with("NULL") { continue; }

        let parts: Vec<&str> = entry.splitn(4, ',').collect();
        if parts.len() < 4 { continue; }

        let name = parts[0].trim().trim_matches('"').to_string();
        if name.is_empty() || name == "NULL" { continue; }

        // Resolve type: look up the macro value, then match the resolved string.
        let type_ref = parts[1].trim();
        let type_value = type_macros.get(type_ref).map(|s| s.as_str()).unwrap_or(type_ref);
        let wtype = if type_value.contains("float") { WEIGHT_TYPE_FLOAT }
            else if type_value.contains("qweight") { WEIGHT_TYPE_QWEIGHT }
            else if type_value.contains("int8") { WEIGHT_TYPE_INT8 }
            else if type_value.contains("int") { WEIGHT_TYPE_INT }
            else { continue };

        let data_name = parts[3].trim().to_string();

        entries.push((name, wtype, 0, data_name));
    }

    entries
}

/// Write a binary blob file from extracted arrays and weight table.
pub fn write_blob(output_path: &str, arrays: &HashMap<String, (i32, Vec<u8>)>,
                  table: &[(String, i32, usize, String)]) -> std::io::Result<usize> {
    let mut f = fs::File::create(output_path)?;
    let mut count = 0;

    for (name, wtype, _declared_size, data_name) in table {
        let Some((_elem_type, data)) = arrays.get(data_name.as_str()) else {
            eprintln!("  Warning: data array '{}' not found for '{}'", data_name, name);
            continue;
        };

        let size = data.len() as i32;
        let block_size = data.len().div_ceil(WEIGHT_BLOCK_SIZE) * WEIGHT_BLOCK_SIZE;

        // Write 64-byte header
        let mut header = [0u8; WEIGHT_BLOCK_SIZE];
        header[0..4].copy_from_slice(b"wght");
        header[4..8].copy_from_slice(&0i32.to_le_bytes());         // version
        header[8..12].copy_from_slice(&wtype.to_le_bytes());       // type
        header[12..16].copy_from_slice(&size.to_le_bytes());       // size
        header[16..20].copy_from_slice(&(block_size as i32).to_le_bytes()); // block_size
        let name_bytes = name.as_bytes();
        let copy_len = name_bytes.len().min(43);
        header[20..20 + copy_len].copy_from_slice(&name_bytes[..copy_len]);
        f.write_all(&header)?;

        // Write data + padding
        f.write_all(data)?;
        let padding = block_size - data.len();
        if padding > 0 {
            f.write_all(&vec![0u8; padding])?;
        }
        count += 1;
    }

    Ok(count)
}

/// Parse a C data file and write the binary blob.
/// `table_name` is the WeightArray variable name (e.g., "fargan_arrays").
pub fn convert_c_to_blob(c_source_path: &str, output_path: &str, table_name: &str) -> std::io::Result<usize> {
    let source = fs::read_to_string(c_source_path)?;
    let arrays = extract_arrays(&source);
    let table = extract_weight_table(&source, table_name);
    if table.is_empty() {
        eprintln!("  No weight table '{}' found in {}", table_name, c_source_path);
        return Ok(0);
    }
    write_blob(output_path, &arrays, &table)
}

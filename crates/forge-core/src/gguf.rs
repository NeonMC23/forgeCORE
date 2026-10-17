//! Safe, payload-lazy inspection of GGUF headers, metadata, and tensor ranges.
//!
//! This module reads only the GGUF header, metadata values, and tensor
//! descriptors. Tensor payload bytes are never copied or decoded. Every
//! descriptor owns its name and dimensions and carries a checked absolute
//! file range, so callers can plan bounded reads without crossing the native
//! API boundary.
//!
//! The contract follows the pinned llama.cpp/ggml GGUF reader: GGUF versions
//! 2 and 3 are accepted; version 1 and newer versions are rejected. Tensor
//! ranks are limited to the pinned `GGML_MAX_DIMS` value of four, tensor names
//! are limited to the pinned 64-byte name buffer, and zero-sized tensors are
//! represented with a zero element and byte count. Tensor offsets must follow
//! the pinned reader's contiguous padded layout, including the final padded
//! data-section extent. Metadata arrays are materialized as Rust-owned
//! non-nested arrays because nested arrays are rejected by the pinned native
//! reader.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::Path;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::quant::{block_spec, tensor_bytes, QuantFormat};

/// GGUF file magic (`GGUF`).
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// Default tensor-data alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// Maximum number of dimensions accepted per tensor descriptor.
pub const MAX_DIMENSIONS: u32 = 4;

const MAX_STRING_LEN: u64 = 16 * 1024 * 1024;
const MAX_METADATA_KEY_LEN: usize = 65_535;
const MAX_TENSOR_NAME_LEN: usize = 64;
const MAX_ARRAY_LEN: u64 = 1_000_000;
const MAX_METADATA_KV: u64 = 100_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_OWNED_BYTES: u64 = 64 * 1024 * 1024;

/// GGUF metadata value type ids.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufValueType {
    /// Unsigned 8-bit integer.
    UInt8 = 0,
    /// Signed 8-bit integer.
    Int8 = 1,
    /// Unsigned 16-bit integer.
    UInt16 = 2,
    /// Signed 16-bit integer.
    Int16 = 3,
    /// Unsigned 32-bit integer.
    UInt32 = 4,
    /// Signed 32-bit integer.
    Int32 = 5,
    /// IEEE binary32 value.
    Float32 = 6,
    /// Boolean encoded as 0 or 1.
    Bool = 7,
    /// UTF-8 string.
    String = 8,
    /// Homogeneous array of a non-array value type.
    Array = 9,
    /// Unsigned 64-bit integer.
    UInt64 = 10,
    /// Signed 64-bit integer.
    Int64 = 11,
    /// IEEE binary64 value.
    Float64 = 12,
}

impl GgufValueType {
    /// Map an on-disk GGUF value type id without invoking native code.
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::UInt8),
            1 => Some(Self::Int8),
            2 => Some(Self::UInt16),
            3 => Some(Self::Int16),
            4 => Some(Self::UInt32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::UInt64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// Rust-owned GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Unsigned 8-bit integer.
    UInt8(u8),
    /// Signed 8-bit integer.
    Int8(i8),
    /// Unsigned 16-bit integer.
    UInt16(u16),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Unsigned 32-bit integer.
    UInt32(u32),
    /// Signed 32-bit integer.
    Int32(i32),
    /// IEEE binary32 value.
    Float32(f32),
    /// Boolean value.
    Bool(bool),
    /// Valid UTF-8 string.
    String(String),
    /// Unsigned 64-bit integer.
    UInt64(u64),
    /// Signed 64-bit integer.
    Int64(i64),
    /// IEEE binary64 value.
    Float64(f64),
    /// Homogeneous non-nested array.
    Array {
        /// Type of each array element.
        element_type: GgufValueType,
        /// Owned array elements.
        values: Vec<MetadataValue>,
    },
}

impl MetadataValue {
    /// Return the GGUF type of this value.
    pub const fn value_type(&self) -> GgufValueType {
        match self {
            Self::UInt8(_) => GgufValueType::UInt8,
            Self::Int8(_) => GgufValueType::Int8,
            Self::UInt16(_) => GgufValueType::UInt16,
            Self::Int16(_) => GgufValueType::Int16,
            Self::UInt32(_) => GgufValueType::UInt32,
            Self::Int32(_) => GgufValueType::Int32,
            Self::Float32(_) => GgufValueType::Float32,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::UInt64(_) => GgufValueType::UInt64,
            Self::Int64(_) => GgufValueType::Int64,
            Self::Float64(_) => GgufValueType::Float64,
            Self::Array { .. } => GgufValueType::Array,
        }
    }

    /// Borrow a string value, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Return the exact GGUF `general.alignment` representation.
    fn alignment_value(&self) -> Option<u64> {
        match self {
            Self::UInt32(value) => Some(u64::from(*value)),
            _ => None,
        }
    }
}

/// A checked, Rust-owned tensor descriptor. The `offset` is relative to the
/// GGUF tensor-data section; `file_offset` is absolute and `byte_length` is
/// the exact validated payload length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Valid non-empty UTF-8 tensor name shorter than 64 bytes.
    pub name: String,
    /// On-disk GGUF dimensions, in `ne[0]..ne[n]` order.
    pub dimensions: Vec<u64>,
    /// Safe-Rust GGML dtype.
    pub dtype: DType,
    /// Checked product of all dimensions.
    pub element_count: u64,
    /// Relative byte offset from the tensor-data section.
    pub offset: u64,
    /// Absolute file byte offset of the tensor payload.
    pub file_offset: u64,
    /// Exact encoded payload length in bytes.
    pub byte_length: u64,
}

impl TensorDescriptor {
    /// Checked exclusive end offset of this tensor's payload.
    pub fn end_offset(&self) -> Result<u64> {
        self.file_offset.checked_add(self.byte_length).ok_or_else(|| {
            Error::gguf(format!(
                "tensor '{}' file range overflows: offset {} + length {}",
                self.name, self.file_offset, self.byte_length
            ))
        })
    }

    /// Return the checked quantization/storage format for this descriptor.
    pub fn quant_format(&self) -> Result<QuantFormat> {
        QuantFormat::from_dtype(self.dtype)
            .map_err(|error| Error::gguf(format!("tensor '{}': {error}", self.name)))
    }
}

/// Payload-lazy GGUF inspection result.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufInspection {
    /// Accepted GGUF format version (2 or 3).
    pub version: u32,
    /// Total source length in bytes.
    pub file_size: u64,
    /// Rust-owned metadata key/value pairs.
    pub metadata: BTreeMap<String, MetadataValue>,
    /// Rust-owned checked tensor descriptors.
    pub tensors: Vec<TensorDescriptor>,
    /// Validated tensor-data alignment.
    pub alignment: u64,
    /// Absolute file offset at which tensor data begins.
    pub data_start: u64,
}

impl GgufInspection {
    /// Inspect a seekable GGUF reader without reading tensor payloads.
    pub fn from_reader<R: Read + Seek>(reader: R) -> Result<Self> {
        inspect(reader)
    }

    /// Inspect an in-memory GGUF byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        inspect_bytes(bytes)
    }

    /// Find a descriptor by its unique GGUF name.
    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Borrow a metadata value by key.
    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }
}

/// Inspect a seekable GGUF reader. Only the header, metadata, and tensor-info
/// records are read; tensor payload bytes are left untouched.
pub fn inspect<R: Read + Seek>(mut reader: R) -> Result<GgufInspection> {
    let file_size = reader
        .seek(SeekFrom::End(0))
        .map_err(|error| Error::gguf(format!("cannot determine input length: {error}")))?;
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| Error::gguf(format!("cannot rewind input: {error}")))?;
    parse(reader, file_size)
}

/// Inspect a filesystem GGUF path without loading tensor payloads.
pub fn inspect_path<P: AsRef<Path>>(path: P) -> Result<GgufInspection> {
    let path = path.as_ref();
    let file = File::open(path)
        .map_err(|error| Error::gguf(format!("cannot open '{}': {error}", path.display())))?;
    inspect(file)
}

/// Inspect an in-memory GGUF byte slice without copying the source payload.
pub fn inspect_bytes(bytes: &[u8]) -> Result<GgufInspection> {
    inspect(Cursor::new(bytes))
}

struct Reader<R> {
    inner: R,
    owned_budget: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            owned_budget: MAX_OWNED_BYTES,
        }
    }

    fn charge(&mut self, amount: u64, what: &str) -> Result<()> {
        if amount > self.owned_budget {
            return Err(Error::gguf(format!(
                "{what} exceeds the bounded inspection allocation budget"
            )));
        }
        self.owned_budget -= amount;
        Ok(())
    }

    fn position(&mut self) -> Result<u64> {
        self.inner
            .stream_position()
            .map_err(|error| Error::gguf(format!("cannot determine GGUF position: {error}")))
    }

    fn read_exact<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        let mut bytes = [0u8; N];
        self.inner.read_exact(&mut bytes).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                Error::gguf(format!("truncated while reading {what}"))
            } else {
                Error::gguf(format!("I/O error while reading {what}: {error}"))
            }
        })?;
        Ok(bytes)
    }

    fn read_bytes(&mut self, len: usize, what: &str) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; len];
        self.inner.read_exact(&mut bytes).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                Error::gguf(format!("truncated while reading {what}"))
            } else {
                Error::gguf(format!("I/O error while reading {what}: {error}"))
            }
        })?;
        Ok(bytes)
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.read_exact::<1>(what)?[0])
    }

    fn u16(&mut self, what: &str) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_exact::<2>(what)?))
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact::<4>(what)?))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact::<8>(what)?))
    }

    fn i8(&mut self, what: &str) -> Result<i8> {
        Ok(i8::from_le_bytes(self.read_exact::<1>(what)?))
    }

    fn i16(&mut self, what: &str) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read_exact::<2>(what)?))
    }

    fn i32(&mut self, what: &str) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_exact::<4>(what)?))
    }

    fn i64(&mut self, what: &str) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_exact::<8>(what)?))
    }

    fn f32(&mut self, what: &str) -> Result<f32> {
        Ok(f32::from_bits(self.u32(what)?))
    }

    fn f64(&mut self, what: &str) -> Result<f64> {
        Ok(f64::from_bits(self.u64(what)?))
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let length = self.u64(&format!("{what} length"))?;
        if length > MAX_STRING_LEN {
            return Err(Error::gguf(format!(
                "{what} length {length} exceeds {MAX_STRING_LEN}"
            )));
        }
        let len = usize::try_from(length).map_err(|_| {
            Error::gguf(format!("{what} length {length} does not fit in usize"))
        })?;
        self.charge(length, what)?;
        let bytes = self.read_bytes(len, what)?;
        String::from_utf8(bytes)
            .map_err(|_| Error::gguf(format!("{what} is not valid UTF-8")))
    }

    fn metadata_value(&mut self, value_type: GgufValueType) -> Result<MetadataValue> {
        match value_type {
            GgufValueType::UInt8 => Ok(MetadataValue::UInt8(self.u8("uint8 metadata")?)),
            GgufValueType::Int8 => Ok(MetadataValue::Int8(self.i8("int8 metadata")?)),
            GgufValueType::UInt16 => Ok(MetadataValue::UInt16(self.u16("uint16 metadata")?)),
            GgufValueType::Int16 => Ok(MetadataValue::Int16(self.i16("int16 metadata")?)),
            GgufValueType::UInt32 => Ok(MetadataValue::UInt32(self.u32("uint32 metadata")?)),
            GgufValueType::Int32 => Ok(MetadataValue::Int32(self.i32("int32 metadata")?)),
            GgufValueType::Float32 => Ok(MetadataValue::Float32(self.f32("float32 metadata")?)),
            GgufValueType::Bool => {
                let value = self.u8("bool metadata")?;
                match value {
                    0 => Ok(MetadataValue::Bool(false)),
                    1 => Ok(MetadataValue::Bool(true)),
                    other => Err(Error::gguf(format!(
                        "boolean metadata value must be 0 or 1, got {other}"
                    ))),
                }
            }
            GgufValueType::String => Ok(MetadataValue::String(self.string("metadata string")?)),
            GgufValueType::Array => self.metadata_array(),
            GgufValueType::UInt64 => Ok(MetadataValue::UInt64(self.u64("uint64 metadata")?)),
            GgufValueType::Int64 => Ok(MetadataValue::Int64(self.i64("int64 metadata")?)),
            GgufValueType::Float64 => Ok(MetadataValue::Float64(self.f64("float64 metadata")?)),
        }
    }

    fn metadata_array(&mut self) -> Result<MetadataValue> {
        let raw_type = self.u32("metadata array element type")?;
        let element_type = GgufValueType::from_raw(raw_type).ok_or_else(|| {
            Error::gguf(format!(
                "unsupported metadata array element type id {raw_type}"
            ))
        })?;
        if element_type == GgufValueType::Array {
            return Err(Error::gguf("nested metadata arrays are unsupported"));
        }
        let count = self.u64("metadata array length")?;
        if count > MAX_ARRAY_LEN {
            return Err(Error::gguf(format!(
                "metadata array length {count} exceeds {MAX_ARRAY_LEN}"
            )));
        }
        let count_usize = usize::try_from(count).map_err(|_| {
            Error::gguf(format!("metadata array length {count} does not fit in usize"))
        })?;
        let vector_bytes = count
            .checked_mul(size_of::<MetadataValue>() as u64)
            .ok_or_else(|| Error::gguf("metadata array allocation size overflows"))?;
        self.charge(vector_bytes, "metadata array")?;
        let mut values = Vec::with_capacity(count_usize);
        for _ in 0..count_usize {
            values.push(self.metadata_value(element_type)?);
        }
        Ok(MetadataValue::Array {
            element_type,
            values,
        })
    }
}

struct RawTensor {
    name: String,
    dimensions: Vec<u64>,
    dtype: DType,
    element_count: u64,
    offset: u64,
}

fn parse<R: Read + Seek>(reader: R, file_size: u64) -> Result<GgufInspection> {
    let mut reader = Reader::new(reader);
    let magic = reader.read_exact::<4>("GGUF magic")?;
    if magic != GGUF_MAGIC {
        return Err(Error::gguf(format!("invalid magic {:?}", magic)));
    }

    let version = reader.u32("GGUF version")?;
    if !matches!(version, 2..=3) {
        return Err(Error::gguf(format!("unsupported GGUF version {version}")));
    }

    let tensor_count = reader.u64("tensor count")?;
    if tensor_count > MAX_TENSORS {
        return Err(Error::gguf(format!(
            "tensor count {tensor_count} exceeds {MAX_TENSORS}"
        )));
    }
    let metadata_count = reader.u64("metadata count")?;
    if metadata_count > MAX_METADATA_KV {
        return Err(Error::gguf(format!(
            "metadata count {metadata_count} exceeds {MAX_METADATA_KV}"
        )));
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.string("metadata key")?;
        if key.is_empty() {
            return Err(Error::gguf("metadata key must not be empty"));
        }
        if key.len() > MAX_METADATA_KEY_LEN || !key.is_ascii() {
            return Err(Error::gguf(format!(
                "metadata key must be ASCII and at most {MAX_METADATA_KEY_LEN} bytes"
            )));
        }
        if metadata.contains_key(&key) {
            return Err(Error::gguf(format!("duplicate metadata key '{key}'")));
        }
        let raw_type = reader.u32("metadata value type")?;
        let value_type = GgufValueType::from_raw(raw_type).ok_or_else(|| {
            Error::gguf(format!("unsupported metadata value type id {raw_type}"))
        })?;
        let value = reader.metadata_value(value_type)?;
        metadata.insert(key, value);
    }

    let alignment = metadata
        .get("general.alignment")
        .map(|value| {
            value.alignment_value().ok_or_else(|| {
                Error::gguf("general.alignment must be encoded as uint32")
            })
        })
        .transpose()?
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::gguf(format!(
            "general.alignment {alignment} must be a non-zero power of two"
        )));
    }

    let mut raw_tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = reader.string("tensor name")?;
        if name.is_empty() {
            return Err(Error::gguf("tensor name must not be empty"));
        }
        if name.len() >= MAX_TENSOR_NAME_LEN {
            return Err(Error::gguf(format!(
                "tensor name length {} is not below pinned GGML_MAX_NAME {MAX_TENSOR_NAME_LEN}",
                name.len()
            )));
        }

        let dimension_count = reader.u32("tensor dimension count")?;
        if dimension_count > MAX_DIMENSIONS {
            return Err(Error::gguf(format!(
                "invalid dimension count {dimension_count}; maximum is {MAX_DIMENSIONS}"
            )));
        }
        let dimensions_len = usize::try_from(dimension_count).map_err(|_| {
            Error::gguf(format!(
                "tensor dimension count {dimension_count} does not fit in usize"
            ))
        })?;
        let dimensions_bytes = u64::from(dimension_count)
            .checked_mul(size_of::<u64>() as u64)
            .ok_or_else(|| Error::gguf("tensor dimensions allocation size overflows"))?;
        reader.charge(dimensions_bytes, "tensor dimensions")?;
        let mut dimensions = Vec::with_capacity(dimensions_len);
        let mut element_count = 1u64;
        for dimension_index in 0..dimensions_len {
            let dimension = reader.u64(&format!("tensor dimension {dimension_index}"))?;
            if dimension > i64::MAX as u64 {
                return Err(Error::gguf(format!(
                    "tensor '{name}' dimension {dimension_index} {dimension} does not fit native int64_t"
                )));
            }
            if element_count != 0 {
                if dimension == 0 {
                    element_count = 0;
                } else {
                    element_count = element_count.checked_mul(dimension).ok_or_else(|| {
                        Error::gguf(format!("tensor '{name}' element count overflows"))
                    })?;
                }
            }
            dimensions.push(dimension);
        }
        if element_count > i64::MAX as u64 {
            return Err(Error::gguf(format!(
                "tensor '{name}' element count {element_count} does not fit native int64_t"
            )));
        }

        let raw_dtype = reader.u32("tensor dtype")?;
        let dtype_id = i32::try_from(raw_dtype).map_err(|_| {
            Error::gguf(format!("unsupported tensor dtype id {raw_dtype}"))
        })?;
        let dtype = DType::from_ggml(dtype_id)
            .map_err(|_| Error::gguf(format!("unsupported tensor dtype id {raw_dtype}")))?;
        let offset = reader.u64("tensor data offset")?;
        reader.charge(size_of::<RawTensor>() as u64, "tensor descriptors")?;
        raw_tensors.push(RawTensor {
            name,
            dimensions,
            dtype,
            element_count,
            offset,
        });
    }

    let mut names = BTreeSet::new();
    for tensor in &raw_tensors {
        if !names.insert(tensor.name.as_str()) {
            return Err(Error::gguf(format!(
                "duplicate tensor name '{}'",
                tensor.name
            )));
        }
    }
    drop(names);

    let tensor_info_end = reader.position()?;
    let data_start = if raw_tensors.is_empty() {
        // The pinned reader does not seek/alignment-pad an empty data section.
        tensor_info_end
    } else {
        align_up(tensor_info_end, alignment)?
    };
    if data_start > file_size {
        return Err(Error::gguf(format!(
            "tensor data start {data_start} is beyond file size {file_size}"
        )));
    }

    let tensor_count = raw_tensors.len();
    let mut tensors = Vec::with_capacity(tensor_count);
    let mut expected_offset = 0u64;
    for (index, raw) in raw_tensors.into_iter().enumerate() {
        if raw.offset != expected_offset {
            return Err(Error::gguf(format!(
                "tensor '{}' has relative offset {}, expected padded offset {}",
                raw.name, raw.offset, expected_offset
            )));
        }

        let format = QuantFormat::from_dtype(raw.dtype).map_err(|_| {
            Error::gguf(format!(
                "tensor '{}' dtype {} is unsupported by the inspection contract",
                raw.name,
                raw.dtype.name()
            ))
        })?;
        let spec = block_spec(format);
        let values_per_block = u64::try_from(spec.values_per_block).map_err(|_| {
            Error::gguf(format!(
                "{} block width does not fit in u64",
                format.name()
            ))
        })?;
        if let Some(&first_dimension) = raw.dimensions.first() {
            if first_dimension % values_per_block != 0 {
                return Err(Error::gguf(format!(
                    "tensor '{}' first dimension {} is not a multiple of {} block values for {}",
                    raw.name,
                    first_dimension,
                    values_per_block,
                    format.name()
                )));
            }
        } else if values_per_block != 1 {
            return Err(Error::gguf(format!(
                "tensor '{}' has no dimension for {} block geometry",
                raw.name,
                format.name()
            )));
        }
        let byte_length = tensor_bytes(format, raw.element_count).map_err(|error| {
            Error::gguf(format!("tensor '{}': {error}", raw.name))
        })?;
        let file_offset = data_start.checked_add(raw.offset).ok_or_else(|| {
            Error::gguf(format!(
                "tensor '{}' absolute offset overflows: data start {} + relative {}",
                raw.name, data_start, raw.offset
            ))
        })?;
        let file_end = file_offset.checked_add(byte_length).ok_or_else(|| {
            Error::gguf(format!(
                "tensor '{}' range overflows: offset {} + length {}",
                raw.name, file_offset, byte_length
            ))
        })?;
        if file_end > file_size {
            return Err(Error::gguf(format!(
                "tensor '{}' range [{file_offset}, {file_end}) exceeds file size {file_size}",
                raw.name
            )));
        }
        let padded_length = align_up(byte_length, alignment)?;
        expected_offset = expected_offset
            .checked_add(padded_length)
            .ok_or_else(|| Error::gguf("padded tensor-data size overflows"))?;

        reader.charge(
            size_of::<TensorDescriptor>() as u64,
            "checked tensor descriptors",
        )?;
        tensors.push(TensorDescriptor {
            name: raw.name,
            dimensions: raw.dimensions,
            dtype: raw.dtype,
            element_count: raw.element_count,
            offset: raw.offset,
            file_offset,
            byte_length,
        });

        // The pinned reader accounts for the padded extent of every tensor,
        // including the final one, before it reads the data section.
        if index + 1 == tensor_count {
            let data_end = data_start.checked_add(expected_offset).ok_or_else(|| {
                Error::gguf("padded tensor-data file range overflows")
            })?;
            if data_end > file_size {
                return Err(Error::gguf(format!(
                    "padded tensor-data range [{data_start}, {data_end}) exceeds file size {file_size}"
                )));
            }
        }
    }

    Ok(GgufInspection {
        version,
        file_size,
        metadata,
        tensors,
        alignment,
        data_start,
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(Error::gguf("alignment must be non-zero"));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| Error::gguf("aligned tensor-data offset overflows"))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor, Read, Seek, SeekFrom};

    use super::*;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &[u8]) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value);
    }

    fn align_vec(bytes: &mut Vec<u8>, alignment: usize) {
        let remainder = bytes.len() % alignment;
        if remainder != 0 {
            bytes.resize(bytes.len() + (alignment - remainder), 0);
        }
    }

    fn make_single_tensor(
        dtype: u32,
        dimensions: &[u64],
        offset: u64,
        payload_len: usize,
    ) -> Vec<u8> {
        make_single_tensor_named(b"tensor", dtype, dimensions, offset, payload_len)
    }

    fn make_single_tensor_named(
        name: &[u8],
        dtype: u32,
        dimensions: &[u64],
        offset: u64,
        payload_len: usize,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 0);
        push_string(&mut bytes, name);
        push_u32(&mut bytes, dimensions.len() as u32);
        for &dimension in dimensions {
            push_u64(&mut bytes, dimension);
        }
        push_u32(&mut bytes, dtype);
        push_u64(&mut bytes, offset);
        align_vec(&mut bytes, DEFAULT_ALIGNMENT as usize);
        if offset <= 1024 {
            bytes.resize(bytes.len() + offset as usize, 0);
            bytes.resize(bytes.len() + payload_len, 0);
        }
        bytes
    }

    fn make_empty_file(version: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut bytes, version);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 0);
        bytes
    }

    fn make_alignment_file(value_type: GgufValueType, value: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 1);
        push_string(&mut bytes, b"general.alignment");
        push_u32(&mut bytes, value_type as u32);
        bytes.extend_from_slice(value);
        bytes
    }

    fn make_two_f32_tensors(second_offset: u64, payload_len: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 2);
        push_u64(&mut bytes, 0);
        for (name, offset) in [(b"first".as_slice(), 0u64), (b"second".as_slice(), second_offset)] {
            push_string(&mut bytes, name);
            push_u32(&mut bytes, 1);
            push_u64(&mut bytes, 1);
            push_u32(&mut bytes, DType::F32.ggml_type() as u32);
            push_u64(&mut bytes, offset);
        }
        align_vec(&mut bytes, DEFAULT_ALIGNMENT as usize);
        bytes.resize(bytes.len() + payload_len, 0);
        bytes
    }

    fn make_valid_with_metadata() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 2);
        push_string(&mut bytes, b"general.architecture");
        push_u32(&mut bytes, GgufValueType::String as u32);
        push_string(&mut bytes, b"llama");
        push_string(&mut bytes, b"general.alignment");
        push_u32(&mut bytes, GgufValueType::UInt32 as u32);
        push_u32(&mut bytes, 32);
        push_string(&mut bytes, b"test.weight");
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 4);
        push_u32(&mut bytes, DType::F32.ggml_type() as u32);
        push_u64(&mut bytes, 0);
        align_vec(&mut bytes, 32);
        // The pinned reader reads the final tensor's padded extent too.
        bytes.extend_from_slice(&[0u8; 32]);
        bytes
    }

    struct HeaderOnlyReader {
        inner: Cursor<Vec<u8>>,
        payload_start: u64,
        max_read_end: u64,
    }

    impl Read for HeaderOnlyReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let start = self.inner.position();
            let end = start
                .checked_add(buffer.len() as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "read position overflow"))?;
            if end > self.payload_start {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "payload read attempted",
                ));
            }
            let count = self.inner.read(buffer)?;
            self.max_read_end = self.max_read_end.max(start + count as u64);
            Ok(count)
        }
    }

    impl Seek for HeaderOnlyReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }

    #[test]
    fn valid_inspection_is_payload_lazy_and_checked() {
        let bytes = make_valid_with_metadata();
        let inspection = inspect_bytes(&bytes).unwrap();
        assert_eq!(inspection.version, 3);
        assert_eq!(inspection.file_size, bytes.len() as u64);
        assert_eq!(inspection.alignment, 32);
        assert_eq!(inspection.metadata["general.architecture"].as_str(), Some("llama"));
        assert_eq!(inspection.tensors.len(), 1);
        let tensor = &inspection.tensors[0];
        assert_eq!(tensor.name, "test.weight");
        assert_eq!(tensor.dimensions, vec![4]);
        assert_eq!(tensor.dtype, DType::F32);
        assert_eq!(tensor.element_count, 4);
        assert_eq!(tensor.offset, 0);
        assert_eq!(tensor.file_offset, inspection.data_start);
        assert_eq!(tensor.byte_length, 16);
        assert_eq!(tensor.end_offset().unwrap(), inspection.data_start + 16);
        assert_eq!(inspection.file_size, inspection.data_start + 32);
        assert_eq!(inspection.tensor("test.weight"), Some(tensor));
    }

    #[test]
    fn metadata_arrays_are_owned_and_nested_arrays_are_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 1);
        push_string(&mut bytes, b"test.values");
        push_u32(&mut bytes, GgufValueType::Array as u32);
        push_u32(&mut bytes, GgufValueType::UInt16 as u32);
        push_u64(&mut bytes, 2);
        bytes.extend_from_slice(&7u16.to_le_bytes());
        bytes.extend_from_slice(&9u16.to_le_bytes());
        let inspection = inspect_bytes(&bytes).unwrap();
        match &inspection.metadata["test.values"] {
            MetadataValue::Array {
                element_type,
                values,
            } => {
                assert_eq!(*element_type, GgufValueType::UInt16);
                assert_eq!(values, &[MetadataValue::UInt16(7), MetadataValue::UInt16(9)]);
            }
            other => panic!("unexpected metadata value: {other:?}"),
        }

        let mut nested = Vec::new();
        nested.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut nested, 3);
        push_u64(&mut nested, 0);
        push_u64(&mut nested, 1);
        push_string(&mut nested, b"nested");
        push_u32(&mut nested, GgufValueType::Array as u32);
        push_u32(&mut nested, GgufValueType::Array as u32);
        push_u64(&mut nested, 0);
        assert!(inspect_bytes(&nested).is_err());
    }

    #[test]
    fn bounded_reader_never_reads_tensor_payload() {
        let bytes = make_valid_with_metadata();
        let expected = inspect_bytes(&bytes).unwrap();
        let mut reader = HeaderOnlyReader {
            inner: Cursor::new(bytes),
            payload_start: expected.data_start,
            max_read_end: 0,
        };
        let inspection = inspect(&mut reader).unwrap();
        assert_eq!(inspection.data_start, expected.data_start);
        assert!(reader.max_read_end <= inspection.data_start);
    }

    #[test]
    fn malformed_magic_and_versions_are_domain_errors() {
        let mut magic = make_valid_with_metadata();
        magic[0] = b'X';
        let error = inspect_bytes(&magic).unwrap_err();
        assert!(error.to_string().starts_with("gguf error:"));

        for version in [1, 4, u32::MAX] {
            assert!(inspect_bytes(&make_empty_file(version)).is_err());
        }
        assert!(inspect_bytes(&make_empty_file(2)).is_ok());
    }

    #[test]
    fn alignment_is_exact_uint32_nonzero_and_power_of_two() {
        let uint64 = make_alignment_file(GgufValueType::UInt64, &32u64.to_le_bytes());
        assert!(inspect_bytes(&uint64).is_err());

        let zero = make_alignment_file(GgufValueType::UInt32, &0u32.to_le_bytes());
        assert!(inspect_bytes(&zero).is_err());
        let non_power_of_two = make_alignment_file(GgufValueType::UInt32, &24u32.to_le_bytes());
        assert!(inspect_bytes(&non_power_of_two).is_err());
        let valid = make_alignment_file(GgufValueType::UInt32, &64u32.to_le_bytes());
        assert_eq!(inspect_bytes(&valid).unwrap().alignment, 64);
    }

    #[test]
    fn truncated_prefixes_never_panic() {
        let bytes = make_valid_with_metadata();
        for length in 0..bytes.len() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                inspect_bytes(&bytes[..length])
            }));
            assert!(result.is_ok(), "parser panicked at prefix length {length}");
            assert!(result.unwrap().is_err(), "prefix {length} unexpectedly parsed");
        }
    }

    #[test]
    fn unsupported_dtype_is_rejected_without_native_calls() {
        let known_but_unsupported = make_single_tensor(3, &[32], 0, 20);
        assert!(inspect_bytes(&known_but_unsupported).is_err());
        let unknown = make_single_tensor(99, &[32], 0, 20);
        assert!(inspect_bytes(&unknown).is_err());
    }

    #[test]
    fn rank_name_and_native_dimension_limits_are_rejected() {
        assert!(inspect_bytes(&make_single_tensor(0, &[1, 1, 1, 1, 1], 0, 0)).is_err());

        let long_name = vec![b'n'; MAX_TENSOR_NAME_LEN];
        assert!(inspect_bytes(&make_single_tensor_named(&long_name, 0, &[1], 0, 0)).is_err());

        let too_large_dimension = make_single_tensor(0, &[i64::MAX as u64 + 1], 0, 0);
        assert!(inspect_bytes(&too_large_dimension).is_err());
    }

    #[test]
    fn zero_sized_tensors_are_owned_and_checked() {
        for dtype in [DType::F32.ggml_type() as u32, DType::Q4_0.ggml_type() as u32] {
            let inspection = inspect_bytes(&make_single_tensor(dtype, &[0], 0, 0)).unwrap();
            let tensor = &inspection.tensors[0];
            assert_eq!(tensor.element_count, 0);
            assert_eq!(tensor.byte_length, 0);
            assert_eq!(tensor.end_offset().unwrap(), tensor.file_offset);
        }
    }

    #[test]
    fn invalid_dimensions_and_padded_offsets_are_rejected() {
        assert!(inspect_bytes(&make_single_tensor(2, &[33], 0, 18)).is_err());

        let valid = make_two_f32_tensors(32, 64);
        let inspection = inspect_bytes(&valid).unwrap();
        assert_eq!(inspection.tensors[0].offset, 0);
        assert_eq!(inspection.tensors[1].offset, 32);

        let gap = make_two_f32_tensors(64, 96);
        assert!(inspect_bytes(&gap).is_err());
        let unaligned = make_two_f32_tensors(16, 64);
        assert!(inspect_bytes(&unaligned).is_err());
    }

    #[test]
    fn bounds_and_checked_overflow_are_rejected() {
        let truncated_payload = make_single_tensor(0, &[4], 0, 15);
        assert!(inspect_bytes(&truncated_payload).is_err());

        let offset_overflow = make_single_tensor(0, &[1], u64::MAX - 31, 4);
        assert!(inspect_bytes(&offset_overflow).is_err());

        let dimension_not_native = make_single_tensor(0, &[u64::MAX, 2], 0, 0);
        assert!(inspect_bytes(&dimension_not_native).is_err());
        let element_count_overflow = make_single_tensor(0, &[i64::MAX as u64, 2], 0, 0);
        assert!(inspect_bytes(&element_count_overflow).is_err());
    }

    #[test]
    fn quantization_geometry_covers_the_inspection_contract() {
        let expected = [
            (QuantFormat::F32, 1, 4),
            (QuantFormat::F16, 1, 2),
            (QuantFormat::BF16, 1, 2),
            (QuantFormat::Q4_0, 32, 18),
            (QuantFormat::Q8_0, 32, 34),
            (QuantFormat::Q2_K, 256, 84),
            (QuantFormat::Q3_K, 256, 110),
            (QuantFormat::Q4_K, 256, 144),
            (QuantFormat::Q5_K, 256, 176),
            (QuantFormat::Q6_K, 256, 210),
            (QuantFormat::Q8_K, 256, 292),
        ];
        for (format, values, bytes) in expected {
            assert_eq!(block_spec(format).values_per_block, values);
            assert_eq!(block_spec(format).bytes_per_block, bytes);
            assert_eq!(tensor_bytes(format, values as u64).unwrap(), bytes as u64);
        }
        assert_eq!(tensor_bytes(QuantFormat::Q4_0, 0).unwrap(), 0);
        assert_eq!(row_bytes(QuantFormat::Q4_0, 0).unwrap(), 0);
        assert!(tensor_bytes(QuantFormat::Q4_0, 33).is_err());
        assert!(tensor_bytes(QuantFormat::Q8_K, u64::MAX - 255).is_err());
    }
}

//! Bounded staging buffers and CUDA transfer primitives.
//!
//! The native layer loads the CUDA driver at runtime. CPU-only systems compile
//! and link this crate normally; creation returns [`StagingError::Unsupported`].
//! Allocations are explicit, handle-based, and never fall back to another
//! memory type.

use std::ffi::CStr;
use std::fmt;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    PageableHost = 1,
    PinnedHost = 2,
    Device = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    HostToDevice = 1,
    DeviceToHost = 2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StagingError {
    InvalidArgument(String),
    InvalidHandle(String),
    Unsupported(String),
    OutOfMemory(String),
    Busy(String),
    NoOperation(String),
    Cuda(String),
    AlreadyReleased(String),
    Internal(String),
}

impl fmt::Display for StagingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message)
            | Self::InvalidHandle(message)
            | Self::Unsupported(message)
            | Self::OutOfMemory(message)
            | Self::Busy(message)
            | Self::NoOperation(message)
            | Self::Cuda(message)
            | Self::AlreadyReleased(message)
            | Self::Internal(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for StagingError {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EngineInfo {
    pub device_index: u32,
    pub device_name: [std::os::raw::c_char; 128],
    pub driver_version: i32,
    pub runtime_version: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BufferInfo {
    pub handle: u64,
    pub memory_type: u32,
    pub state: u32,
    pub capacity: u64,
    pub allocated: u64,
    pub in_flight: u8,
    pub host_pinned: u8,
    pub device_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub allocation_requests: u64,
    pub allocation_successes: u64,
    pub allocation_failures: u64,
    pub released: u64,
    pub current_bytes: u64,
    pub high_water_bytes: u64,
    pub transfer_requests: u64,
    pub transfer_completed: u64,
    pub transfer_failed: u64,
    pub transfer_bytes: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub last_duration_ns: u64,
    pub total_duration_ns: u64,
    pub in_flight_operations: u64,
    pub cancellation_requests: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TransferReport {
    pub bytes: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub duration_ns: u64,
    pub completed: u32,
    pub direction: u32,
}

impl TransferReport {
    pub fn bandwidth_gbps(&self) -> Option<f64> {
        if self.bytes == 0 || self.duration_ns == 0 {
            None
        } else {
            Some(self.bytes as f64 / (self.duration_ns as f64 / 1e9) / 1e9)
        }
    }
}

impl Default for EngineInfo {
    fn default() -> Self {
        Self {
            device_index: 0,
            device_name: [0; 128],
            driver_version: 0,
            runtime_version: 0,
        }
    }
}

impl EngineInfo {
    pub fn device_name(&self) -> String {
        unsafe { CStr::from_ptr(self.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

mod ffi {
    use super::*;
    use std::os::raw::{c_char, c_int};

    extern "C" {
        pub fn fstage_engine_create(device_index: u32, out_engine: *mut u64) -> c_int;
        pub fn fstage_engine_destroy(engine: u64) -> c_int;
        pub fn fstage_engine_get_info(engine: u64, out: *mut EngineInfo) -> c_int;
        pub fn fstage_engine_stats(engine: u64, out: *mut Stats) -> c_int;
        pub fn fstage_buffer_create(
            engine: u64,
            memory_type: u32,
            bytes: u64,
            out_buffer: *mut u64,
        ) -> c_int;
        pub fn fstage_buffer_release(buffer: u64) -> c_int;
        pub fn fstage_buffer_get_info(buffer: u64, out: *mut BufferInfo) -> c_int;
        pub fn fstage_buffer_write(
            buffer: u64,
            offset: u64,
            data: *const std::ffi::c_void,
            size: u64,
        ) -> c_int;
        pub fn fstage_buffer_read(
            buffer: u64,
            offset: u64,
            out: *mut std::ffi::c_void,
            size: u64,
        ) -> c_int;
        pub fn fstage_transfer_submit(
            engine: u64,
            src: u64,
            src_offset: u64,
            dst: u64,
            dst_offset: u64,
            bytes: u64,
            direction: u32,
        ) -> c_int;
        pub fn fstage_transfer_submit_to_ptr(
            engine: u64,
            src: u64,
            src_offset: u64,
            device_ptr: u64,
            bytes: u64,
        ) -> c_int;
        pub fn fstage_transfer_query(engine: u64, out: *mut TransferReport) -> c_int;
        pub fn fstage_transfer_wait(engine: u64, out: *mut TransferReport) -> c_int;
        pub fn fstage_transfer_cancel(engine: u64) -> c_int;
        pub fn fstage_last_error(out: *mut c_char, size: u64) -> c_int;
    }
}

fn last_error() -> String {
    let mut buffer = [0i8; 512];
    unsafe {
        ffi::fstage_last_error(buffer.as_mut_ptr(), buffer.len() as u64);
        CStr::from_ptr(buffer.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

fn map_error(code: i32) -> StagingError {
    let message = last_error();
    match code {
        1 => StagingError::InvalidArgument(message),
        2 => StagingError::InvalidHandle(message),
        3 => StagingError::Unsupported(message),
        4 => StagingError::OutOfMemory(message),
        5 => StagingError::Busy(message),
        6 => StagingError::NoOperation(message),
        7 => StagingError::Cuda(message),
        9 => StagingError::AlreadyReleased(message),
        _ => StagingError::Internal(message),
    }
}

fn check(code: i32) -> Result<(), StagingError> {
    if code == 0 {
        Ok(())
    } else {
        Err(map_error(code))
    }
}

pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    handle: u64,
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        let code = unsafe { ffi::fstage_engine_destroy(self.handle) };
        if code != 0 {
            eprintln!(
                "forge-staging: engine destroy failed ({}); resources remain owned",
                last_error()
            );
        }
    }
}

impl Engine {
    pub fn new(device_index: u32) -> Result<Self, StagingError> {
        let mut handle = 0u64;
        check(unsafe { ffi::fstage_engine_create(device_index, &mut handle) })?;
        Ok(Self {
            inner: Arc::new(EngineInner { handle }),
        })
    }

    pub fn info(&self) -> Result<EngineInfo, StagingError> {
        let mut info = EngineInfo::default();
        check(unsafe { ffi::fstage_engine_get_info(self.inner.handle, &mut info) })?;
        Ok(info)
    }

    pub fn stats(&self) -> Result<Stats, StagingError> {
        let mut stats = Stats::default();
        check(unsafe { ffi::fstage_engine_stats(self.inner.handle, &mut stats) })?;
        Ok(stats)
    }

    pub fn allocate(
        &self,
        memory_type: MemoryType,
        bytes: u64,
    ) -> Result<StagingBuffer, StagingError> {
        let mut handle = 0u64;
        check(unsafe {
            ffi::fstage_buffer_create(self.inner.handle, memory_type as u32, bytes, &mut handle)
        })?;
        Ok(StagingBuffer {
            engine: Arc::clone(&self.inner),
            handle,
            context: memory_type,
        })
    }

    pub fn submit(
        &self,
        source: &StagingBuffer,
        source_offset: u64,
        destination: &StagingBuffer,
        destination_offset: u64,
        bytes: u64,
        direction: Direction,
    ) -> Result<(), StagingError> {
        check(unsafe {
            ffi::fstage_transfer_submit(
                self.inner.handle,
                source.handle,
                source_offset,
                destination.handle,
                destination_offset,
                bytes,
                direction as u32,
            )
        })
    }

    /// H2D copy into an externally owned CUDA device pointer. The pointer must
    /// belong to the same primary context/device; ownership is not transferred.
    pub fn submit_to_device_ptr(
        &self,
        source: &StagingBuffer,
        source_offset: u64,
        device_ptr: u64,
        bytes: u64,
    ) -> Result<(), StagingError> {
        check(unsafe {
            ffi::fstage_transfer_submit_to_ptr(
                self.inner.handle,
                source.handle,
                source_offset,
                device_ptr,
                bytes,
            )
        })
    }

    pub fn query(&self) -> Result<Option<TransferReport>, StagingError> {
        let mut report = TransferReport::default();
        let code = unsafe { ffi::fstage_transfer_query(self.inner.handle, &mut report) };
        if code == 6 {
            return Ok(None);
        }
        check(code)?;
        Ok((report.completed != 0).then_some(report))
    }

    pub fn wait(&self) -> Result<TransferReport, StagingError> {
        let mut report = TransferReport::default();
        check(unsafe { ffi::fstage_transfer_wait(self.inner.handle, &mut report) })?;
        Ok(report)
    }

    pub fn cancel(&self) -> Result<(), StagingError> {
        check(unsafe { ffi::fstage_transfer_cancel(self.inner.handle) })
    }

    #[cfg(test)]
    fn raw_handle(&self) -> u64 {
        self.inner.handle
    }
}

pub struct StagingBuffer {
    engine: Arc<EngineInner>,
    handle: u64,
    context: MemoryType,
}

impl StagingBuffer {
    pub fn memory_type(&self) -> MemoryType {
        self.context
    }

    #[cfg(test)]
    fn raw_handle(&self) -> u64 {
        self.handle
    }

    pub fn info(&self) -> Result<BufferInfo, StagingError> {
        let mut info = BufferInfo::default();
        check(unsafe { ffi::fstage_buffer_get_info(self.handle, &mut info) })?;
        Ok(info)
    }

    pub fn write_host(&self, offset: u64, data: &[u8]) -> Result<(), StagingError> {
        check(unsafe {
            ffi::fstage_buffer_write(self.handle, offset, data.as_ptr().cast(), data.len() as u64)
        })
    }

    pub fn read_host(&self, offset: u64, out: &mut [u8]) -> Result<(), StagingError> {
        check(unsafe {
            ffi::fstage_buffer_read(
                self.handle,
                offset,
                out.as_mut_ptr().cast(),
                out.len() as u64,
            )
        })
    }

    /// Explicit release. After success the buffer is unusable; Drop is a no-op.
    /// A busy buffer can be retried after its transfer completes.
    pub fn release(&mut self) -> Result<(), StagingError> {
        check(unsafe { ffi::fstage_buffer_release(self.handle) })?;
        self.handle = 0;
        Ok(())
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        let code = unsafe { ffi::fstage_buffer_release(self.handle) };
        if code != 0 {
            // Fail closed: never free/reuse memory that may still be referenced.
            eprintln!(
                "forge-staging: buffer release failed ({}); leaking handle rather than reusing it",
                last_error()
            );
            self.handle = 0;
        }
        // Keep the engine Arc alive until after this drop.
        let _ = &self.engine;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_arguments_are_typed() {
        let engine = match Engine::new(0) {
            Ok(engine) => engine,
            Err(StagingError::Unsupported(_)) => {
                println!("SKIP: CUDA driver unavailable");
                return;
            }
            Err(error) => panic!("unexpected engine error: {error}"),
        };
        assert!(engine.allocate(MemoryType::PageableHost, 0).is_err());
        let mut buffer = engine.allocate(MemoryType::PageableHost, 16).unwrap();
        assert!(buffer.write_host(0, &[0u8; 32]).is_err());
        assert!(buffer.read_host(0, &mut [0u8; 32]).is_err());
        let mut device = engine.allocate(MemoryType::Device, 16).unwrap();
        assert!(buffer.write_host(0, &[0u8; 16]).is_ok());
        assert!(device.write_host(0, &[0u8; 16]).is_err());
        buffer.release().unwrap();
        device.release().unwrap();
    }

    #[test]
    fn pinned_and_device_roundtrip_has_measured_bytes() {
        let engine = match Engine::new(0) {
            Ok(engine) => engine,
            Err(StagingError::Unsupported(_)) => {
                println!("SKIP: CUDA driver unavailable");
                return;
            }
            Err(error) => panic!("unexpected engine error: {error}"),
        };
        let bytes = 1024 * 1024u64;
        let mut host = engine.allocate(MemoryType::PinnedHost, bytes).unwrap();
        let mut device = engine.allocate(MemoryType::Device, bytes).unwrap();
        let source: Vec<u8> = (0..bytes)
            .map(|i| (i.wrapping_mul(31) & 0xff) as u8)
            .collect();
        host.write_host(0, &source).unwrap();
        engine
            .submit(&host, 0, &device, 0, bytes, Direction::HostToDevice)
            .unwrap();
        let h2d = engine.wait().unwrap();
        assert_eq!(h2d.bytes, bytes);
        assert!(h2d.duration_ns > 0);
        assert!(h2d.bandwidth_gbps().unwrap() > 0.0);

        let mut readback = vec![0u8; source.len()];
        let mut destination = engine.allocate(MemoryType::PinnedHost, bytes).unwrap();
        engine
            .submit(&device, 0, &destination, 0, bytes, Direction::DeviceToHost)
            .unwrap();
        let d2h = engine.wait().unwrap();
        destination.read_host(0, &mut readback).unwrap();
        assert_eq!(readback, source);
        assert_eq!(d2h.bytes, bytes);

        let stats = engine.stats().unwrap();
        assert_eq!(stats.allocation_successes, 3);
        assert_eq!(stats.transfer_completed, 2);
        assert_eq!(stats.transfer_failed, 0);
        assert_eq!(stats.transfer_bytes, bytes * 2);
        assert_eq!(stats.in_flight_operations, 0);
        assert!(stats.high_water_bytes >= bytes * 2);
        host.release().unwrap();
        device.release().unwrap();
        destination.release().unwrap();
        assert_eq!(engine.stats().unwrap().current_bytes, 0);
    }

    #[test]
    fn release_while_in_flight_is_rejected_and_destroy_with_buffer_is_busy() {
        let engine = match Engine::new(0) {
            Ok(engine) => engine,
            Err(StagingError::Unsupported(_)) => {
                println!("SKIP: CUDA driver unavailable");
                return;
            }
            Err(error) => panic!("unexpected engine error: {error}"),
        };
        let bytes = 4 * 1024 * 1024u64;
        let mut host = engine.allocate(MemoryType::PinnedHost, bytes).unwrap();
        let mut device = engine.allocate(MemoryType::Device, bytes).unwrap();
        host.write_host(0, &vec![1u8; bytes as usize]).unwrap();
        engine
            .submit(&host, 0, &device, 0, bytes, Direction::HostToDevice)
            .unwrap();
        assert!(matches!(host.release(), Err(StagingError::Busy(_))));
        let report = engine.wait().unwrap();
        assert_eq!(report.bytes, bytes);
        // Explicit destroy is refused while buffers remain allocated.
        let code = unsafe { ffi::fstage_engine_destroy(engine.raw_handle()) };
        assert_eq!(code, 5, "engine destroy with buffers must be BUSY");
        assert!(engine.cancel().is_err());
        host.release().unwrap();
        device.release().unwrap();
    }

    #[test]
    fn allocation_failure_and_duplicate_release_are_typed() {
        let engine = match Engine::new(0) {
            Ok(engine) => engine,
            Err(StagingError::Unsupported(_)) => {
                println!("SKIP: CUDA driver unavailable");
                return;
            }
            Err(error) => panic!("unexpected engine error: {error}"),
        };
        let error = match engine.allocate(MemoryType::Device, 1u64 << 50) {
            Ok(_) => panic!("1 PiB device allocation must fail"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            StagingError::OutOfMemory(_) | StagingError::Cuda(_)
        ));
        let mut buffer = engine.allocate(MemoryType::PageableHost, 64).unwrap();
        let handle = buffer.raw_handle();
        buffer.release().unwrap();
        let code = unsafe { ffi::fstage_buffer_release(handle) };
        assert_eq!(code, 9, "duplicate release must be ALREADY_RELEASED");
    }

    #[test]
    fn cancellation_during_transfer_is_explicitly_unsafe_to_cancel() {
        let engine = match Engine::new(0) {
            Ok(engine) => engine,
            Err(StagingError::Unsupported(_)) => {
                println!("SKIP: CUDA driver unavailable");
                return;
            }
            Err(error) => panic!("unexpected engine error: {error}"),
        };
        let bytes = 64 * 1024 * 1024u64;
        let mut host = engine.allocate(MemoryType::PinnedHost, bytes).unwrap();
        let mut device = engine.allocate(MemoryType::Device, bytes).unwrap();
        host.write_host(0, &vec![3u8; bytes as usize]).unwrap();
        engine
            .submit(&host, 0, &device, 0, bytes, Direction::HostToDevice)
            .unwrap();
        let error = engine.cancel().unwrap_err();
        assert!(matches!(error, StagingError::Busy(_)));
        let _ = engine.wait().unwrap();
        host.release().unwrap();
        device.release().unwrap();
    }

    #[test]
    fn pageable_transfer_is_available_and_measured() {
        let engine = match Engine::new(0) {
            Ok(engine) => engine,
            Err(StagingError::Unsupported(_)) => {
                println!("SKIP: CUDA driver unavailable");
                return;
            }
            Err(error) => panic!("unexpected engine error: {error}"),
        };
        let bytes = 256 * 1024u64;
        let mut host = engine.allocate(MemoryType::PageableHost, bytes).unwrap();
        let mut device = engine.allocate(MemoryType::Device, bytes).unwrap();
        host.write_host(0, &vec![7u8; bytes as usize]).unwrap();
        engine
            .submit(&host, 0, &device, 0, bytes, Direction::HostToDevice)
            .unwrap();
        let report = engine.wait().unwrap();
        assert_eq!(report.bytes, bytes);
        assert!(report.duration_ns > 0);
        host.release().unwrap();
        device.release().unwrap();
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;

    fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        sorted[(sorted.len() * numerator / denominator).min(sorted.len() - 1)]
    }

    fn run_size(engine: &Engine, memory_type: MemoryType, bytes: u64, label: &str) {
        let mut host = engine.allocate(memory_type, bytes).unwrap();
        let mut device = engine.allocate(MemoryType::Device, bytes).unwrap();
        let mut readback = engine.allocate(MemoryType::PinnedHost, bytes).unwrap();
        host.write_host(0, &vec![0x5a; bytes as usize]).unwrap();
        // Warm up both directions.
        engine
            .submit(&host, 0, &device, 0, bytes, Direction::HostToDevice)
            .unwrap();
        let _ = engine.wait().unwrap();
        engine
            .submit(&device, 0, &readback, 0, bytes, Direction::DeviceToHost)
            .unwrap();
        let _ = engine.wait().unwrap();

        let mut h2d = Vec::new();
        let mut d2h = Vec::new();
        let mut submit = Vec::new();
        for _ in 0..5 {
            engine
                .submit(&host, 0, &device, 0, bytes, Direction::HostToDevice)
                .unwrap();
            let report = engine.wait().unwrap();
            h2d.push(report.duration_ns);
            submit.push(report.submit_ns);
            engine
                .submit(&device, 0, &readback, 0, bytes, Direction::DeviceToHost)
                .unwrap();
            let report = engine.wait().unwrap();
            d2h.push(report.duration_ns);
        }
        h2d.sort_unstable();
        d2h.sort_unstable();
        submit.sort_unstable();
        let median_h2d = percentile(&h2d, 1, 2);
        let median_d2h = percentile(&d2h, 1, 2);
        let median_submit = percentile(&submit, 1, 2);
        println!(
            "{} {} bytes={} H2D_med_ns={} H2D_p25_ns={} H2D_p75_ns={} D2H_med_ns={} D2H_p95_ns={} submit_med_ns={} H2D_GBps={:.2} D2H_GBps={:.2}",
            label,
            if memory_type == MemoryType::PinnedHost { "pinned" } else { "pageable" },
            bytes,
            median_h2d,
            percentile(&h2d, 1, 4),
            percentile(&h2d, 3, 4),
            median_d2h,
            percentile(&d2h, 95, 100),
            median_submit,
            bytes as f64 / (median_h2d as f64 / 1e9) / 1e9,
            bytes as f64 / (median_d2h as f64 / 1e9) / 1e9,
        );
        host.release().unwrap();
        device.release().unwrap();
        readback.release().unwrap();
    }

    #[test]
    #[ignore = "physical CUDA transfer benchmark"]
    fn transfer_benchmark_matrix() {
        let engine = match Engine::new(0) {
            Ok(engine) => engine,
            Err(error) => {
                eprintln!("SKIP: {error}");
                return;
            }
        };
        println!("engine: {:?}", engine.info().expect("info").device_name());
        for (label, bytes) in [
            ("smollm_layer", 2_596_608u64),
            ("medium", 16 * 1024 * 1024),
            ("large", 64 * 1024 * 1024),
            ("qwen_layer", 183 * 1024 * 1024),
        ] {
            run_size(&engine, MemoryType::PinnedHost, bytes, label);
            if bytes <= 64 * 1024 * 1024 {
                run_size(&engine, MemoryType::PageableHost, bytes, label);
            }
        }
        let started = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = engine.stats().unwrap();
        }
        println!("snapshot_overhead_1000_ns={}", started.elapsed().as_nanos());
    }
}

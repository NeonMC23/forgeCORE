//! Isolated, file-scoped residency-transition validation.
//!
//! This module exists to validate what `mincore()` can and cannot prove. It
//! creates a dedicated temporary file, maps it, touches known pages, observes
//! residency, requests a file-scoped cache drop with `posix_fadvise`, and
//! observes again. No global cache dropping and no memory-pressure stimulus is
//! used.
//!
//! The experiment never claims that a page was physically read, that a page
//! fault occurred, that I/O happened, or that a kernel accessed a page. A
//! reported transition means only that `mincore()` reported resident pages
//! before a request and nonresident pages afterwards.

use std::fmt;

/// Upper bound for one experiment. Keeps auxiliary buffers bounded.
pub const MAX_EXPERIMENT_PAGES: u64 = 4096;
/// Upper bound for transition cycles in one experiment.
pub const MAX_EXPERIMENT_CYCLES: u32 = 16;

/// Page counts from one `mincore()` observation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidencyCounts {
    pub requested_pages: u64,
    pub sampled_pages: u64,
    pub resident_pages: u64,
    pub nonresident_pages: u64,
    pub unknown_pages: u64,
}

/// Outcome of one requested resident-to-nonresident transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionOutcome {
    /// Every requested page was resident before and nonresident after.
    Observed,
    /// Every requested page was still resident after the eviction request.
    NotObservedStillResident,
    /// The observation mixed resident/nonresident/unknown pages.
    Partial,
    /// The platform or kernel did not support the observation path.
    Unsupported,
    /// The experiment could not be prepared or observed.
    Failed,
}

impl fmt::Display for TransitionOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Observed => "observed",
            Self::NotObservedStillResident => "not-observed-still-resident",
            Self::Partial => "partial",
            Self::Unsupported => "unsupported",
            Self::Failed => "failed",
        })
    }
}

/// One touch/observe/evict/observe/re-touch/observe cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyExperimentCycle {
    pub cycle: u32,
    /// Observation after the experiment touched every page.
    pub after_touch: ResidencyCounts,
    /// Observation after unmapping, the file-scoped cache request, and
    /// remapping the same file.
    pub after_eviction: ResidencyCounts,
    /// Observation after touching every page again.
    pub after_retouch: ResidencyCounts,
    pub unmap_ok: bool,
    pub fadvise_ok: bool,
    pub remap_ok: bool,
    pub eviction_errno: Option<i32>,
    pub outcome: TransitionOutcome,
}

/// Storage backing used for the experiment file. `Tmpfs` is reported
/// separately because file-scoped cache eviction requests generally cannot
/// make tmpfs pages nonresident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStorage {
    Tmpfs,
    Filesystem,
    Unknown,
}

/// Structured result of a controlled residency experiment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyExperimentReport {
    /// Always `controlled-file-mapping`, not a production model mapping.
    pub preparation: &'static str,
    /// Monotonic identity for this isolated experiment mapping.
    pub mapping_generation: u64,
    /// The observation and eviction requests actually used.
    pub method: &'static str,
    pub platform: &'static str,
    pub storage: ExperimentStorage,
    pub page_size: u64,
    pub requested_pages: u64,
    pub cycles: Vec<ResidencyExperimentCycle>,
    pub transition_observed: bool,
    pub outcome: TransitionOutcome,
    pub notes: Vec<String>,
}

/// Configuration for a controlled experiment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyExperimentConfig {
    pub pages: u64,
    pub cycles: u32,
    /// Directory for the temporary experiment file. `None` uses
    /// `FORGE_RESIDENCY_EXPERIMENT_DIR` when set, otherwise the process
    /// temporary directory.
    pub directory: Option<std::path::PathBuf>,
}

impl Default for ResidencyExperimentConfig {
    fn default() -> Self {
        Self {
            pages: 16,
            cycles: 1,
            directory: None,
        }
    }
}

impl ResidencyExperimentConfig {
    fn validate(self) -> Result<Self, ResidencyExperimentError> {
        if self.pages == 0 || self.pages > MAX_EXPERIMENT_PAGES {
            return Err(ResidencyExperimentError::InvalidPages(self.pages));
        }
        if self.cycles == 0 || self.cycles > MAX_EXPERIMENT_CYCLES {
            return Err(ResidencyExperimentError::InvalidCycles(self.cycles));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidencyExperimentError {
    InvalidPages(u64),
    InvalidCycles(u32),
    Unsupported(&'static str),
    Io(String),
    Mapping(String),
    Observation(String),
}

impl fmt::Display for ResidencyExperimentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPages(pages) => write!(
                formatter,
                "experiment pages must be in 1..={MAX_EXPERIMENT_PAGES}, got {pages}"
            ),
            Self::InvalidCycles(cycles) => write!(
                formatter,
                "experiment cycles must be in 1..={MAX_EXPERIMENT_CYCLES}, got {cycles}"
            ),
            Self::Unsupported(reason) => {
                write!(formatter, "residency experiment unsupported: {reason}")
            }
            Self::Io(reason) => write!(formatter, "residency experiment I/O error: {reason}"),
            Self::Mapping(reason) => {
                write!(formatter, "residency experiment mapping error: {reason}")
            }
            Self::Observation(reason) => {
                write!(
                    formatter,
                    "residency experiment observation error: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for ResidencyExperimentError {}

/// Run a controlled, file-scoped residency experiment.
///
/// On Linux this creates a unique temporary file under the process temporary
/// directory and removes it before returning. Other platforms report
/// [`ResidencyExperimentError::Unsupported`].
pub fn run_file_residency_experiment(
    config: ResidencyExperimentConfig,
) -> Result<ResidencyExperimentReport, ResidencyExperimentError> {
    let config = config.validate()?;
    platform_run(config)
}

#[cfg(target_os = "linux")]
mod platform {
    use std::ffi::c_void;
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::os::fd::AsRawFd;
    use std::path::PathBuf;
    use std::ptr;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        ExperimentStorage, ResidencyCounts, ResidencyExperimentConfig, ResidencyExperimentCycle,
        ResidencyExperimentError, ResidencyExperimentReport, TransitionOutcome,
    };

    static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
    static EXPERIMENT_GENERATION: AtomicU64 = AtomicU64::new(0);
    const METHOD: &str =
        "mmap(MAP_SHARED) + touch + mincore + munmap + posix_fadvise(DONTNEED) + mmap + mincore";
    const PREPARATION: &str = "controlled-file-mapping";
    const PLATFORM: &str = "linux";

    struct TempMapping {
        file: File,
        path: PathBuf,
        addr: *mut c_void,
        len: usize,
    }

    impl TempMapping {
        fn create(
            pages: u64,
            page_size: u64,
            directory: &std::path::Path,
        ) -> Result<Self, ResidencyExperimentError> {
            let len = pages
                .checked_mul(page_size)
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or_else(|| {
                    ResidencyExperimentError::Mapping("mapping length overflow".to_string())
                })?;
            let unique = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0);
            std::fs::create_dir_all(directory)
                .map_err(|error| ResidencyExperimentError::Io(error.to_string()))?;
            let path = directory.join(format!(
                "forge-residency-{}-{nanos}-{unique}.bin",
                std::process::id()
            ));
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| ResidencyExperimentError::Io(error.to_string()))?;
            file.set_len(len as u64)
                .map_err(|error| ResidencyExperimentError::Io(error.to_string()))?;
            file.write_all(&[0]).ok();
            let addr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                let error = std::io::Error::last_os_error().to_string();
                let _ = std::fs::remove_file(&path);
                return Err(ResidencyExperimentError::Mapping(error));
            }
            Ok(Self {
                file,
                path,
                addr,
                len,
            })
        }

        fn touch_pages(&self, pages: u64, page_size: u64) {
            // SAFETY: the mapping covers `pages * page_size` bytes, and each
            // offset below is within that mapping. The write faults the page
            // into the page cache but is not evidence of a physical read.
            unsafe {
                for index in 0..pages {
                    let offset = (index * page_size) as usize;
                    ptr::write_volatile((self.addr as *mut u8).add(offset), 1);
                }
                let _ = libc::msync(self.addr, self.len, libc::MS_SYNC);
            }
        }

        fn sample(&self, page_size: u64) -> Result<ResidencyCounts, ResidencyExperimentError> {
            sample_mapping(self.addr, self.len, page_size)
        }

        fn unmap(&mut self) -> bool {
            if self.addr.is_null() {
                return false;
            }
            // SAFETY: the mapping is live and is unmapped exactly once here.
            let ok = unsafe { libc::munmap(self.addr, self.len) == 0 };
            self.addr = ptr::null_mut();
            ok
        }

        fn evict_file_cache(&self) -> (bool, Option<i32>) {
            // File-scoped request only. It does not drop system-wide cache and
            // is advisory: the following observation decides whether anything
            // actually changed.
            let fadvise_result = unsafe {
                libc::posix_fadvise(
                    self.file.as_raw_fd(),
                    0,
                    self.len as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                )
            };
            (
                fadvise_result == 0,
                (fadvise_result != 0).then_some(fadvise_result),
            )
        }

        fn remap(&mut self) -> Result<(), ResidencyExperimentError> {
            let addr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    self.len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.file.as_raw_fd(),
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                return Err(ResidencyExperimentError::Mapping(
                    std::io::Error::last_os_error().to_string(),
                ));
            }
            self.addr = addr;
            Ok(())
        }
    }

    impl Drop for TempMapping {
        fn drop(&mut self) {
            // SAFETY: this mapping was created with mmap and is unmapped once.
            unsafe {
                let _ = libc::munmap(self.addr, self.len);
            }
            let _ = std::fs::remove_file(&self.path);
        }
    }

    pub(super) fn run(
        config: ResidencyExperimentConfig,
    ) -> Result<ResidencyExperimentReport, ResidencyExperimentError> {
        let page_size = page_size()?;
        let mapping_generation = EXPERIMENT_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
        let directory = config
            .directory
            .clone()
            .or_else(|| std::env::var_os("FORGE_RESIDENCY_EXPERIMENT_DIR").map(Into::into))
            .unwrap_or_else(std::env::temp_dir);
        let storage = storage_kind(&directory);
        let mut mapping = TempMapping::create(config.pages, page_size, &directory)?;
        let mut cycles = Vec::with_capacity(config.cycles as usize);
        let mut transition_observed = false;
        let mut notes = Vec::new();
        let mut eviction_never_observed = true;

        for cycle in 1..=config.cycles {
            mapping.touch_pages(config.pages, page_size);
            let after_touch = mapping.sample(page_size)?;
            let unmap_ok = mapping.unmap();
            let (fadvise_ok, eviction_errno) = mapping.evict_file_cache();
            mapping.remap()?;
            let after_eviction = mapping.sample(page_size)?;
            mapping.touch_pages(config.pages, page_size);
            let after_retouch = mapping.sample(page_size)?;

            let outcome = classify(config.pages, &after_touch, &after_eviction);
            transition_observed |= outcome == TransitionOutcome::Observed;
            eviction_never_observed &= outcome != TransitionOutcome::Observed;
            cycles.push(ResidencyExperimentCycle {
                cycle,
                after_touch,
                after_eviction,
                after_retouch,
                unmap_ok,
                fadvise_ok,
                remap_ok: true,
                eviction_errno,
                outcome,
            });
        }

        if storage == ExperimentStorage::Tmpfs {
            notes.push(
                "The experiment file is on tmpfs; file-scoped fadvise generally cannot make \
                 tmpfs pages nonresident. Use a disk-backed directory for transition validation."
                    .to_string(),
            );
        }
        if eviction_never_observed {
            notes.push(
                "The kernel did not report a full resident-to-nonresident transition for this \
                 file mapping. The eviction request is advisory; no transition is claimed."
                    .to_string(),
            );
        }
        notes.push("the temporary file and mapping are released before return".to_string());
        notes.push(
            "Residency observations are not evidence of physical reads, page faults, I/O, or \
             kernel access."
                .to_string(),
        );

        let outcome = if transition_observed {
            TransitionOutcome::Observed
        } else if cycles
            .iter()
            .all(|cycle| cycle.outcome == TransitionOutcome::Unsupported)
        {
            TransitionOutcome::Unsupported
        } else if cycles
            .iter()
            .any(|cycle| cycle.outcome == TransitionOutcome::Partial)
        {
            TransitionOutcome::Partial
        } else {
            TransitionOutcome::NotObservedStillResident
        };

        Ok(ResidencyExperimentReport {
            preparation: PREPARATION,
            mapping_generation,
            method: METHOD,
            platform: PLATFORM,
            storage,
            page_size,
            requested_pages: config.pages,
            cycles,
            transition_observed,
            outcome,
            notes,
        })
    }

    fn storage_kind(directory: &std::path::Path) -> ExperimentStorage {
        let Ok(path) = std::ffi::CString::new(directory.as_os_str().as_encoded_bytes()) else {
            return ExperimentStorage::Unknown;
        };
        let mut stats: libc::statfs = unsafe { std::mem::zeroed() };
        // SAFETY: `path` is NUL-terminated and `stats` is a valid out pointer.
        if unsafe { libc::statfs(path.as_ptr(), &mut stats) } != 0 {
            return ExperimentStorage::Unknown;
        }
        const TMPFS_MAGIC: libc::c_long = 0x0102_1994;
        if stats.f_type == TMPFS_MAGIC {
            ExperimentStorage::Tmpfs
        } else {
            ExperimentStorage::Filesystem
        }
    }

    fn page_size() -> Result<u64, ResidencyExperimentError> {
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        u64::try_from(size)
            .ok()
            .filter(|size| *size > 0)
            .ok_or_else(|| ResidencyExperimentError::Observation("invalid page size".to_string()))
    }

    fn sample_mapping(
        addr: *mut c_void,
        len: usize,
        page_size: u64,
    ) -> Result<ResidencyCounts, ResidencyExperimentError> {
        let pages = expected_pages(addr, len, page_size)?;
        let mut states = vec![0u8; pages];
        sample_mapping_into(addr, len, page_size, &mut states)
    }

    fn expected_pages(
        addr: *mut c_void,
        len: usize,
        page_size: u64,
    ) -> Result<usize, ResidencyExperimentError> {
        if addr.is_null() || len == 0 {
            return Err(ResidencyExperimentError::Observation(
                "empty mapping".to_string(),
            ));
        }
        let pages = len / page_size as usize;
        if pages == 0 {
            return Err(ResidencyExperimentError::Observation(
                "mapping shorter than one page".to_string(),
            ));
        }
        Ok(pages)
    }

    fn sample_mapping_into(
        addr: *mut c_void,
        len: usize,
        page_size: u64,
        states: &mut [u8],
    ) -> Result<ResidencyCounts, ResidencyExperimentError> {
        let pages = expected_pages(addr, len, page_size)?;
        if states.len() < pages {
            return Err(ResidencyExperimentError::Observation(
                "page-state buffer is too small".to_string(),
            ));
        }
        // SAFETY: `states` has at least `len / page_size` bytes, as required
        // by mincore, and `addr`/`len` describe a live mapping.
        let result = unsafe { libc::mincore(addr, len, states.as_mut_ptr()) };
        if result != 0 {
            return Err(ResidencyExperimentError::Observation(format!(
                "mincore failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        let mut counts = ResidencyCounts {
            requested_pages: pages as u64,
            sampled_pages: pages as u64,
            ..ResidencyCounts::default()
        };
        for state in states.iter().take(pages) {
            if state & 1 != 0 {
                counts.resident_pages += 1;
            } else {
                counts.nonresident_pages += 1;
            }
        }
        Ok(counts)
    }

    fn classify(
        pages: u64,
        after_touch: &ResidencyCounts,
        after_eviction: &ResidencyCounts,
    ) -> TransitionOutcome {
        if after_touch.resident_pages == pages && after_eviction.nonresident_pages == pages {
            TransitionOutcome::Observed
        } else if after_eviction.resident_pages == pages
            && after_eviction.unknown_pages == 0
            && after_eviction.nonresident_pages == 0
        {
            TransitionOutcome::NotObservedStillResident
        } else {
            TransitionOutcome::Partial
        }
    }

    #[cfg(test)]
    mod mapping_probe_tests {
        use super::*;

        #[test]
        fn invalid_or_missing_mapping_is_an_error_not_nonresident() {
            let page_size = page_size().expect("page size");
            assert!(matches!(
                sample_mapping(std::ptr::null_mut(), page_size as usize, page_size),
                Err(ResidencyExperimentError::Observation(_))
            ));
            assert!(matches!(
                sample_mapping(std::ptr::null_mut(), 0, page_size),
                Err(ResidencyExperimentError::Observation(_))
            ));
        }

        #[test]
        fn a_released_mapping_is_not_reported_as_nonresident() {
            let page_size = page_size().expect("page size");
            let mut mapping =
                TempMapping::create(1, page_size, &std::env::temp_dir()).expect("mapping");
            let address = mapping.addr;
            let length = mapping.len;
            // Allocate the state buffer before unmapping so the allocator
            // cannot reuse the released address during the probe.
            let mut states = vec![0u8; 1];
            assert!(mapping.unmap());
            let error = sample_mapping_into(address, length, page_size, &mut states)
                .expect_err("unmapped probe");
            assert!(matches!(error, ResidencyExperimentError::Observation(_)));
        }
    }
}

#[cfg(target_os = "linux")]
fn platform_run(
    config: ResidencyExperimentConfig,
) -> Result<ResidencyExperimentReport, ResidencyExperimentError> {
    platform::run(config)
}

#[cfg(not(target_os = "linux"))]
fn platform_run(
    _config: ResidencyExperimentConfig,
) -> Result<ResidencyExperimentReport, ResidencyExperimentError> {
    Err(ResidencyExperimentError::Unsupported(
        "controlled file residency experiments currently require Linux mincore/posix_fadvise",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk_backed_test_directory() -> Option<std::path::PathBuf> {
        let candidate = std::path::PathBuf::from("/var/tmp");
        candidate
            .is_dir()
            .then(|| candidate.join("forge-residency-experiments"))
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        assert!(matches!(
            run_file_residency_experiment(ResidencyExperimentConfig {
                pages: 0,
                cycles: 1,
                directory: None,
            }),
            Err(ResidencyExperimentError::InvalidPages(0))
        ));
        assert!(matches!(
            run_file_residency_experiment(ResidencyExperimentConfig {
                pages: 1,
                cycles: 0,
                directory: None,
            }),
            Err(ResidencyExperimentError::InvalidCycles(0))
        ));
        assert!(matches!(
            run_file_residency_experiment(ResidencyExperimentConfig {
                pages: MAX_EXPERIMENT_PAGES + 1,
                cycles: 1,
                directory: None,
            }),
            Err(ResidencyExperimentError::InvalidPages(_))
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn touch_is_observed_as_resident_and_report_is_consistent() {
        let report = run_file_residency_experiment(ResidencyExperimentConfig {
            pages: 8,
            cycles: 1,
            directory: disk_backed_test_directory(),
        })
        .expect("experiment");
        assert_eq!(report.requested_pages, 8);
        assert_eq!(report.platform, "linux");
        assert_eq!(report.preparation, "controlled-file-mapping");
        let cycle = &report.cycles[0];
        assert_eq!(cycle.after_touch.requested_pages, 8);
        assert_eq!(cycle.after_touch.sampled_pages, 8);
        assert_eq!(cycle.after_touch.resident_pages, 8);
        assert_eq!(cycle.after_touch.nonresident_pages, 0);
        assert_eq!(cycle.after_touch.unknown_pages, 0);
        assert_eq!(cycle.after_retouch.resident_pages, 8);
        assert_eq!(
            report.transition_observed,
            cycle.outcome == TransitionOutcome::Observed
        );
        if !report.transition_observed {
            assert!(report
                .notes
                .iter()
                .any(|note| note.contains("no transition is claimed")));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn disk_backed_transition_is_claimed_only_from_observation() {
        let report = run_file_residency_experiment(ResidencyExperimentConfig {
            pages: 8,
            cycles: 2,
            directory: disk_backed_test_directory(),
        })
        .expect("experiment");
        if report.storage != ExperimentStorage::Filesystem {
            eprintln!("SKIP: no disk-backed experiment directory available");
            return;
        }
        assert!(report.transition_observed, "{report:?}");
        assert_eq!(report.outcome, TransitionOutcome::Observed);
        for cycle in &report.cycles {
            assert_eq!(cycle.after_touch.resident_pages, 8);
            assert_eq!(cycle.after_eviction.nonresident_pages, 8);
            assert_eq!(cycle.after_retouch.resident_pages, 8);
            assert!(cycle.unmap_ok && cycle.fadvise_ok && cycle.remap_ok);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn concurrent_experiments_use_isolated_files_and_mappings() {
        let directory = disk_backed_test_directory();
        let handles: Vec<_> = (0..4)
            .map(|index| {
                let directory = directory.clone();
                std::thread::spawn(move || {
                    run_file_residency_experiment(ResidencyExperimentConfig {
                        pages: 4 + index,
                        cycles: 1,
                        directory,
                    })
                    .expect("concurrent experiment")
                })
            })
            .collect();
        for (index, handle) in handles.into_iter().enumerate() {
            let report = handle.join().expect("experiment thread");
            assert_eq!(report.requested_pages, 4 + index as u64);
            assert_eq!(report.cycles.len(), 1);
            assert_eq!(
                report.cycles[0].after_touch.resident_pages,
                4 + index as u64
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn repeated_cycles_re_observe_instead_of_reusing_stale_values() {
        let report = run_file_residency_experiment(ResidencyExperimentConfig {
            pages: 4,
            cycles: 3,
            directory: disk_backed_test_directory(),
        })
        .expect("experiment");
        assert_eq!(report.cycles.len(), 3);
        for (index, cycle) in report.cycles.iter().enumerate() {
            assert_eq!(cycle.cycle, index as u32 + 1);
            assert_eq!(cycle.after_touch.resident_pages, 4);
            assert_eq!(cycle.after_retouch.resident_pages, 4);
            assert_eq!(cycle.after_touch.requested_pages, 4);
        }
    }
}

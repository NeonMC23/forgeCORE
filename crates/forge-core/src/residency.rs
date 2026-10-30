//! Native resource-lifetime facts and release errors.
//!
//! These types describe what the pinned native implementation can prove.
//! They do not authorize intra-decode eviction: a synchronized context still
//! has logical model weights that the next token will reuse. They do make the
//! safe between-operation boundary enforceable, including a non-blocking
//! `Busy` result when asynchronous graph work is still in flight.

use crate::error::Error;
use std::fmt;

/// Version of the native residency/ownership contract compiled into
/// `forge-sys`. This must match the native build; it is not a model format.
pub const RESIDENCY_CONTRACT_VERSION: u32 = 1;

/// Process-local model lifetime facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLifetimeState {
    /// Stable for this loaded model instance. Never zero for a live model.
    pub instance_id: u64,
    pub active_contexts: u64,
    pub inflight_contexts: u64,
    /// True only when an exclusive quiescence check succeeded at the
    /// observation instant. Not a promise about future operations.
    pub quiescent: bool,
}

/// Per-context execution facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLifetimeState {
    pub context_id: u64,
    pub execution_generation: u64,
    pub inflight: bool,
    pub synchronized: bool,
}

/// Explicit failure categories for native residency operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidencyError {
    /// A context still has asynchronous graph work in flight. The native
    /// call did not block and did not release anything.
    Busy,
    /// The native platform or build does not support this operation.
    Unsupported,
    /// The request was malformed.
    Invalid(String),
    /// Native errno not otherwise classified.
    Native(i32),
}

const EBUSY_ERRNO: i32 = 16;
#[cfg(target_os = "macos")]
const ENOTSUP_ERRNO: i32 = 45;
#[cfg(not(target_os = "macos"))]
const ENOTSUP_ERRNO: i32 = 95;

impl ResidencyError {
    pub fn from_native_errno(errno: i32) -> Self {
        if errno == EBUSY_ERRNO {
            Self::Busy
        } else if errno == ENOTSUP_ERRNO {
            Self::Unsupported
        } else {
            Self::Native(errno)
        }
    }
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => formatter
                .write_str("resource still in use: context has asynchronous compute in flight"),
            Self::Unsupported => formatter.write_str("native residency operation is unsupported"),
            Self::Invalid(message) => write!(formatter, "invalid residency request: {message}"),
            Self::Native(errno) => write!(formatter, "native residency failure with errno {errno}"),
        }
    }
}

impl std::error::Error for ResidencyError {}

impl From<ResidencyError> for Error {
    fn from(error: ResidencyError) -> Self {
        Error::model(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errno_classification_is_explicit() {
        assert_eq!(ResidencyError::from_native_errno(16), ResidencyError::Busy);
        assert_eq!(
            ResidencyError::from_native_errno(if cfg!(target_os = "macos") { 45 } else { 95 }),
            ResidencyError::Unsupported
        );
        assert_eq!(
            ResidencyError::from_native_errno(22),
            ResidencyError::Native(22)
        );
    }
}

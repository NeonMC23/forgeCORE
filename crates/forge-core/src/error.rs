//! Single error type for ForgeCore.
//!
//! Every fallible operation returns [`Result<T, Error>`](crate::Result).
//! There are no panics on the numerical path for shape or configuration
//! problems: arity mismatches, invalid dimensions, unsupported formats,
//! and backend failures are explicit errors.

use std::fmt;

/// Explicit ForgeCore failure (validation or backend error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl Error {
    /// Backend / FFI failure (allocation, execution, model loading, ...).
    pub fn backend(message: impl Into<String>) -> Self {
        Self(format!("backend error: {}", message.into()))
    }

    /// Well-formed request ForgeCore cannot satisfy (yet).
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self(format!("unsupported: {}", message.into()))
    }

    /// Malformed request (bad shape, arity, dimensions, ...).
    pub fn invalid(message: impl Into<String>) -> Self {
        Self(format!("invalid: {}", message.into()))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Fallible result used across ForgeCore.
pub type Result<T> = std::result::Result<T, Error>;

//! Single error type for ForgeCore reference kernels.
//!
//! Every fallible operation returns [`Result<T, Error>`](crate::Result).
//! There are no panics on the numerical path for shape or configuration
//! problems: arity mismatches, invalid dimensions, and unsupported formats
//! are explicit errors.

use std::fmt;

/// Explicit numerical-contract failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Fallible result used by every ForgeCore kernel.
pub type Result<T> = std::result::Result<T, Error>;

//! Single error type for ForgeCore.
//!
//! Every fallible operation returns [`Result<T, Error>`](crate::Result).
//! There are no panics on the numerical path for shape or configuration
//! problems: arity mismatches, invalid dimensions, unsupported formats,
//! and backend failures are explicit errors.
//!
//! Constructors prefix messages by failure domain so callers (and
//! RAMforge, later) can distinguish where a failure came from without
//! matching on free-form text.

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

    /// Model loading or model metadata failure.
    pub fn model(message: impl Into<String>) -> Self {
        Self(format!("model error: {}", message.into()))
    }

    /// Context creation or context use failure.
    pub fn context(message: impl Into<String>) -> Self {
        Self(format!("context error: {}", message.into()))
    }

    /// Batch construction or batch validation failure.
    pub fn batch(message: impl Into<String>) -> Self {
        Self(format!("batch error: {}", message.into()))
    }

    /// Decode execution failure (native error codes map here).
    pub fn decode(message: impl Into<String>) -> Self {
        Self(format!("decode error: {}", message.into()))
    }

    /// Logits access failure.
    pub fn logits(message: impl Into<String>) -> Self {
        Self(format!("logits error: {}", message.into()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_prefix_by_domain() {
        assert_eq!(Error::backend("x").0, "backend error: x");
        assert_eq!(Error::unsupported("x").0, "unsupported: x");
        assert_eq!(Error::invalid("x").0, "invalid: x");
        assert_eq!(Error::model("x").0, "model error: x");
        assert_eq!(Error::context("x").0, "context error: x");
        assert_eq!(Error::batch("x").0, "batch error: x");
        assert_eq!(Error::decode("x").0, "decode error: x");
        assert_eq!(Error::logits("x").0, "logits error: x");
    }
}

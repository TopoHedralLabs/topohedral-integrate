//! Error used by one-shot operations that combine configuration and integration.

//{{{ crate imports
use crate::{config::ConfigError, gauss::RuleError, integration::IntegrationError};
//}}}
//{{{ dep imports
use thiserror::Error;
//}}}

//{{{ enum: OptionsError
/// Error returned by an operation that configures and performs integration in one call.
///
/// See the [adaptive one-shot guide](https://topohedrallabs.github.io/topohedral-integrate/latest/user-guide/adaptive-quadrature/#one-shot-helpers).
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OptionsError {
    /// A Gaussian rule could not be constructed.
    #[error(transparent)]
    Rule(#[from] RuleError),
    /// A validated configuration could not be constructed.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// An integrand could not be evaluated.
    #[error(transparent)]
    Integration(#[from] IntegrationError),
}
//}}}

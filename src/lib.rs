//! This crate provides a collection of quadature rules for numerical integration along with
//! algorithms for adaptive integration build on top of these rules.
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]

mod common;
pub use common::{OptionsError, OptionsStruct};
pub mod adaptive;
pub mod fixed;
pub mod gauss;

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {
    use ctor::ctor;
    use topohedral_tracing::*;

    #[ctor]
    fn init_logger() {
        init().unwrap();
    }

    #[test]
    fn test_logging() {
        info!("Logging is working!");
    }
}
//}}}

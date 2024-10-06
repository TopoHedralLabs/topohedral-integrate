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




mod common;
pub use common::{OptionsError, OptionsStruct};
pub mod gauss;
pub mod adaptive;
pub mod fixed;




//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use ctor::ctor;
    use topohedral_tracing::*;
    use super::*;

    #[ctor]
    fn init_logger() {
        init().unwrap();
    }

    #[test]
    fn test_logging() 
    {
        info!("Logging is working!");
    }


}
//}}}
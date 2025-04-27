//! Common definitions and utilities used throughout the crate.   
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------

/// Simple trait which all options structs must implement.
pub trait OptionsStruct {
    /// Checks if the options are valid.
    ///
    /// If the options are valid, returns `Ok(())`. Else, returns `Err(OptionsError)` with the
    /// respective enum variant. If `full` is set to false then only the cheap version of the
    /// check is performed, meaning it only says if it is valid or not with no diagnostic
    /// information. If `full` is set to true then the full version of the check is performed,
    /// meaning every error is reposred in the string contained by InvalidOptionsFull.  
    fn is_ok(&self, full: bool) -> Result<(), OptionsError>;
}

#[derive(Error, Debug)]
pub enum OptionsError {
    #[error("The options are invalid.")]
    InvalidOptionsShort,
    #[error("The options are invalid with reasons:{0}")]
    InvalidOptionsFull(String),
}

/// Appends the reason to the error.
pub fn append_reason(err: &mut OptionsError, reason: &str) {
    match err {
        OptionsError::InvalidOptionsShort => {}
        OptionsError::InvalidOptionsFull(s) => {
            s.push_str(format!("\n\t{}", reason).as_str());
        }
    }
}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_error_append() {
        let mut err = OptionsError::InvalidOptionsFull(String::new());
        append_reason(&mut err, "reason 1");
        append_reason(&mut err, "reason 2");

        let err_str = format!("{}", err);
        assert_eq!(
            err_str,
            "The options are invalid with reasons:\n\treason 1\n\treason 2"
        );
    }
}
//}}}

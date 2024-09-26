
pub mod gauss;
pub mod adaptive;


//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use ctor::ctor;
    use topohedral_tracing::*;

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
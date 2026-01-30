class BoostCorrError(Exception):
    """Base class for all exceptions in this project."""
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


class ComputingDeviceError(BoostCorrError):
    """Specific to computing device."""
    def __init__(self, message="Computing device error"):
        super().__init__(message, exit_code=70)


class PackageImportError(BoostCorrError):
    """Specific to package import."""
    def __init__(self, message="Package import error"):
        super().__init__(message, exit_code=71)


class QMapError(BoostCorrError):
    """Specific to qmap."""
    def __init__(self, message="Qmap error"):
        super().__init__(message, exit_code=72)


class DatasetError(BoostCorrError):
    """Specific to dataset."""
    def __init__(self, message="Dataset error"):
        super().__init__(message, exit_code=73)


class CorrelatorError(BoostCorrError):
    """Specific to correlator."""
    def __init__(self, message="Correlator error"):
        super().__init__(message, exit_code=74)


class ProcessingError(BoostCorrError):
    """Specific to processing."""
    def __init__(self, message="Processing error"):
        super().__init__(message, exit_code=75)


class ResultSavingError(BoostCorrError):
    """Specific to result saving."""
    def __init__(self, message="Result saving error"):
        super().__init__(message, exit_code=76)


class PostProcessingError(BoostCorrError):
    """Specific to post processing."""
    def __init__(self, message="Post processing error"):
        super().__init__(message, exit_code=77)


class ConfigurationError(BoostCorrError):
    """Specific to configuration and arguments."""
    def __init__(self, message="Configuration error"):
        super().__init__(message, exit_code=78)


class InputError(BoostCorrError):
    """Specific to input data or parameters."""
    def __init__(self, message="Input error"):
        super().__init__(message, exit_code=65)
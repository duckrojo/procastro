"""
Custom exceptions for the API Service.
These exceptions are used to provide more specific error handling for the different API services.
"""

class ApiServiceError(Exception):
    """Base exception for all API Service errors"""
    def __init__(self, message="API Service error occurred", details=None, provider=None):
        self.message = message
        self.details = details or {}
        self.provider = provider
        super().__init__(self.message)
    
    def __str__(self):
        base_msg = f"{self.message}"
        if self.provider:
            base_msg += f" [Provider: {self.provider}]"
        if self.details:
            base_msg += f" - Details: {self.details}"
        return base_msg


class HttpProviderError(ApiServiceError):
    """Exception raised for HTTP provider errors"""
    def __init__(self, message="HTTP request failed", status_code=None, url=None, **kwargs):
        details = kwargs.get("details", {})
        if status_code:
            details["status_code"] = status_code
        if url:
            details["url"] = url
        super().__init__(message=message, details=details, provider="HttpProvider")


class AstroqueryProviderError(ApiServiceError):
    """Exception raised for Astroquery provider errors"""
    def __init__(self, message="Astroquery operation failed", query_object=None, **kwargs):
        details = kwargs.get("details", {})
        if query_object:
            details["query_object"] = query_object
        super().__init__(message=message, details=details, provider="AstroqueryProvider")


class SimbadProviderError(AstroqueryProviderError):
    """Exception raised for SIMBAD provider errors"""
    def __init__(self, message="SIMBAD query failed", **kwargs):
        super().__init__(message=message, provider="SimbadProvider", **kwargs)


class ExoplanetProviderError(AstroqueryProviderError):
    """Exception raised for Exoplanet Archive provider errors"""
    def __init__(self, message="Exoplanet Archive query failed", **kwargs):
        super().__init__(message=message, provider="ExoplanetProvider", **kwargs)


class LocalFilesProviderError(ApiServiceError):
    """Exception raised for local files provider errors"""
    def __init__(self, message="Local file operation failed", file_path=None, **kwargs):
        details = kwargs.get("details", {})
        if file_path:
            details["file_path"] = file_path
        super().__init__(message=message, details=details, provider="LocalFilesProvider")


class DataValidationError(ApiServiceError):
    """Exception raised when data validation fails"""
    def __init__(self, message="Data validation failed", validation_errors=None, **kwargs):
        details = kwargs.get("details", {})
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message=message, details=details, **kwargs)
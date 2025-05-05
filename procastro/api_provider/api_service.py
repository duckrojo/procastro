


import logging
import time
from typing import Any, Callable, Dict, Optional

import requests


logger = logging.getLogger(__name__)

import pyvo as vo


class ApiError(Exception):
    """Base exception for API errors"""
    pass


class ApiService:
    """
    A class to handle API request and response processing.
    It can be used to request data from Horizons and queries from databases in raw sql.
    """



    def __init__(self):
        """Initialize the API service with default configuration."""
        # Default base URLs for common services
        self.base_urls = {
            'horizons': 'https://ssd.jpl.nasa.gov/api/horizons.api',
            'usgs_maps': 'https://astrogeology.usgs.gov/maps/',
            'tap': 'https://exoplanetarchive.ipac.caltech.edu/TAP'
        }
        
        # Request configuration
        self.timeout = (5, 30)  # (connect_timeout, read_timeout)
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # Headers
        self.default_headers = {
            'User-Agent': 'procastro/1.0 (Astronomical Python Package)',
        }


    def get(self, url: str, params: Optional[Dict] = None, 
            service: Optional[str] = None, **kwargs) -> requests.Response:
        """
        Send a GET request to the specified URL.
        
        Parameters
        ----------
        url : str
            URL to send request to. If service is provided, this can be a relative path.
        params : Dict, optional
            Query parameters for the request
        service : str, optional
            Service name to use base URL from self.base_urls
        **kwargs : dict
            Additional parameters to pass to requests.get
            
        Returns
        -------
        requests.Response
            Response object
            
        Raises
        ------
        ConnectionError
            If connection fails after retries
        TimeoutError
            If request times out after retries
        RequestError
            For other request errors
        """
        full_url = self._build_url(url, service)
        
        # Merge default headers with any provided headers
        headers = self.default_headers.copy()
        if 'headers' in kwargs:
            headers.update(kwargs.pop('headers'))
        
        # Set default timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
            
        logger.debug(f"GET request to {full_url}")
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(full_url, params=params, headers=headers, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.ConnectionError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Connection error after {self.max_retries} attempts: {str(e)}")
                    raise ConnectionError(f"Failed to connect to {full_url}: {str(e)}")
                logger.warning(f"Connection error (attempt {attempt+1}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
            except requests.exceptions.Timeout as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Timeout error after {self.max_retries} attempts: {str(e)}")
                    raise TimeoutError(f"Request to {full_url} timed out: {str(e)}")
                logger.warning(f"Timeout error (attempt {attempt+1}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during GET request: {str(e)}")
                raise

    def _handle_error(self, error: Exception, error_type: str, url: str, 
                     fallback: Optional[Callable] = None, 
                     return_none: bool = False) -> Optional[Any]:
        """
        Handle errors from API requests.
        
        Parameters
        ----------
        error : Exception
            The exception that occurred
        error_type : str
            Type of error (for logging)
        url : str
            URL that was requested
        fallback : Callable, optional
            Function to call with the error
        return_none : bool
            Whether to return None instead of raising an exception
            
        Returns
        -------
        Optional[Any]
            Result of fallback function if provided, None if return_none is True
            
        Raises
        ------
        ApiError
            If no fallback is provided and return_none is False
        """
        error_msg = f"{error_type} when requesting {url}: {str(error)}"
        logger.error(error_msg)
        
        if fallback:
            logger.info(f"Using fallback for {url}")
            return fallback(error)
            
        if return_none:
            return None
            
        raise ApiError(error_msg) from error


    def tap_service(self,url= "https://exoplanetarchive.ipac.caltech.edu/TAP", query=str) :
        """
        query the exoplanet archive using the TAP service.
        Parameters
        ----------
        url : str
            URL of the TAP service. Default is the Exoplanet Archive TAP service.
        query : str
            SQL query to execute on
        
        """
        service = vo.dal.TAPService(url)
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        return service.search(query)



    
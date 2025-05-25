"""
API that encapsulates all possible services used in PROCASTRO.
The idea behind this API is to take advantage of the class hierarchy to create a single API that can be used to access all services.
For example, we have a DataProvider abstract class, which is above all the Transport methods (HTTP, SQL, localfiles)
Then, we have a DataProvider implementation for each transport method (HTTP, SQL, LOCALFILES)
Finally, we hace a Provider for every service (SIMBAD, USGS, HORIZONS, ETC).
This way, adding a provider is very easy, and can be done in a single file.
"""



from abc import ABC, abstractmethod
from functools import wraps
import logging
import os
import time
from typing import Callable, Generic, Optional, TypeVar
import astroquery.simbad as aqs
import pandas as pd
import requests
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


# first, we configure the logging system
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# type for results
T = TypeVar("T")  # this means that the result can be any type


class ApiResult(Generic[T]):
    """
    Generic container class, it will hold data of type T and T will be determined when the class is instantiated.
    
    Example usage:
    ```python
        result = api.GET(service=API.SIMBAD, object="M31")
        if result.success:
            # Use the data with full type hinting support
            star_data = result.data
        else:
            print(f"Error: {result.error}")
    ```
    """

    def __init__(self,
                data: T = None,
                success: bool = False,
                error: str = None,
                source: str = None,
                is_fallback: bool = False
            ):  
        self.data = data # the actual data returned
        self.success = success # whether the request was successful or not
        self.error = error # error message if the request failed
        self.source = source # which provider returned this data
        self.is_fallback = is_fallback #whether this came from a fallback source 
    
class DataProviderInterface(ABC):
    """
    Abstract class that defines the interface that all data providers must implement.
    """

    @abstractmethod
    def request(self, **kwargs) -> ApiResult:
        """
        Method that will be used to request data from the provider.
        It will return an ApiResult object.
        """
        pass


    @abstractmethod
    def support_params(self) -> list:
        """
        Method that will return a list of parameters that the provider supports.
        """
        pass
    # NOTE: These two method are not specified, because they need to be implemented in the concrete classes.
    
    @staticmethod
    def with_fallback(fallback_func: Optional[Callable] = None,
                    return_empty_on_fail: bool = False) -> Callable:
        """
        Method that will set a fallback function for the provider.
        This function will be called if the original method fails.
        Args:
            fallback_func: function to call if the original method fails.
            return_empty_on_fail: whether to return an empty result if the original method fails. If it's set to True,
                the method wont raise and exception, but it will return an empty result, otherwise, it will raise an exception.
            On every case, the fallback function will always be called if provided. 
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, **kwargs):
                try:
                    # Try original method
                    return func(self, **kwargs)
                except Exception as e:
                    logger.error(f"{self.__class__.__name__} error: {e}")
                    
                    # Try fallback if provided
                    if fallback_func:
                        try:
                            fallback_data = fallback_func(self,e, **kwargs)
                            return ApiResult(
                                data=fallback_data, 
                                success=True,
                                source=f"{self.__class__.__name__}_fallback",
                                is_fallback=True
                            )
                        except Exception as fallback_error:
                            raise Exception(f"Fallback function failed: {fallback_error}") from e
                    
                    # Return empty result if configured
                    if return_empty_on_fail:
                        return ApiResult(
                            data=None, 
                            success=False,
                            error=e,
                            source=self.__class__.__name__
                        )
                    
                    # Re-raise if no fallback handling succeeded
                    raise
            return wrapper
        return decorator


class HttpProvider(DataProviderInterface):
    """
    Provider that will help to instantiate HTTP Providers . USGS y HORIZONS.
    """


    # NOTE: Here, we can set the set of parameters that this type of providers support.
    def support_params(self) -> list:
        """
        Method that will return a list of parameters that the provider supports.
        """
        return ["url", "headers", "params", "data", "timeout", "verbose","method","json","cookies", "files", "auth","allow_redirects","proxies","verify","stream","cert"]
    
    # NOTE: Decorators on parent classes are "overrided" as long as we dont call the super() method, otherwise, this decorator will be called on runtime.
    @DataProviderInterface.with_fallback(return_empty_on_fail=True)
    def request(self,  **kwargs)-> ApiResult:
        """
        Method that handles HTTP requests
        """
        url = kwargs.get("url")
        method = kwargs.get("method", "GET")
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 1)
        verbose = kwargs.pop("verbose", False)# we pop verbose argument since it's not used by the request class.|
    
        
        if verbose: logger.info(f"HTTP request: {method} {url}")
        # NOTE: Here we use the request library to make the request.
        if not url:
            return ApiResult(
                data=None,
                success=False,
                error="URL is required",
                source=self.__class__.__name__
            )
        

        kwargs = {key: value for key, value in kwargs.items() if key in self.support_params()}
        if "method" not in kwargs:
            logger.warning("Method not specified, defaulting to GET")
            kwargs["method"] = "GET"
        # When we do the request, there are 3 cases.
        # 1. The request is successful (2xx) and we get a response.
        # 2. The request is not successful (4xx, 5xx) and we get an error. In this case, we will NOT raise an exception.
        # 3. The server is not reachable (timeout, connection error, etc). In this case, we will raise an exception.
        for attempt in range(max_retries):
            try:
                # Make the request
                response = requests.request(**kwargs)
                response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
                
                if verbose : logger.info(f"HTTP response: {response.status_code} ")
                # Handle different respose types:
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    data = response.json()
                elif any(binary_type in content_type for binary_type in [
                    "image/", "application/octet-stream", "application/pdf", 
                    "application/zip", "binary/"
                ]):
                    # Return raw binary content for images and other binary types
                    data = response.content
                else:
                    data = response.text
                return ApiResult(
                    data=data,
                    success=True,
                    source=self.__class__.__name__)
            except requests.HTTPError as e:
                status_code = e.response.status_code
                logger.error(f"HTTP error: {status_code} - for {url} - Error: {e}")

                if 400 <= status_code < 500:
                    return ApiResult(
                        data=None,
                        success=False,
                        error=f"Client error: {url}",
                        source=self.__class__.__name__
                    )
                elif 500 <= status_code < 600:
                    return ApiResult(
                        data=None,
                        success=False,
                        error=f"Server error: {url}",
                        source=self.__class__.__name__
                    )
            except (requests.Timeout, requests.ConnectionError) as e: 
                if attempt < max_retries - 1:
                    logger.error(f"Request error: {url} - Error: {e}, retrying...")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                raise Exception(f"Request failed after {max_retries} retries: Timeout")
            except requests.RequestException as e:
                logger.error(f"Request error: {url} - Error: {e}")
                raise Exception(f"Request failed: {e}") 
            except requests.TooManyRedirects as e:
                logger.error(f"Too many redirects: {url} - Error: {e}")
                raise Exception(f"Too many redirects: {e}")

class AstroqueryProvider(DataProviderInterface):
    """
    Provider which handles astronomical queries using ASTROQUERY (like TAP, SIMBAD or others)

    """
    @abstractmethod
    def support_params(self):
        """
        Common parameters for all astronomical queries using astroquery.
        We will be implementing this method on its child classes.
        """
        pass


class LocalFilesProvider(DataProviderInterface):
    """
    Class to use when loading local files
    """

    def __init__(self, api_service=None):
        self.api_service = api_service
    
    def set_api_service(self, api_service):
        """Set the API service reference after initialization"""
        self.api_service = api_service
    def support_params(self):
        return ["file_path", "force_reload", "reload_days"]
    
    def exoplanet_fallback(self, error, **kwargs):

        if not hasattr(self, 'api_service') or self.api_service is None:
            # Si no tenemos la referencia, creamos una nueva instancia
            # (esto es menos eficiente pero funciona como fallback)
            from procastro.api_provider.api_service import ApiService
            api_service = ApiService()
        else:
            api_service = self.api_service

        result: ApiResult = api_service.query_exoplanet(
            table="pscomppars",
            select="pl_name,ra,dec,pl_orbper,pl_tranmid,disc_facility,pl_trandur,sy_pmra,sy_pmdec,sy_vmag,sy_gmag",
            where="pl_tranmid!=0.0 and pl_orbper!=0.0"
        )
        if not result.success:
            raise Exception("API call failed and fallback also failed")

        return result

    @DataProviderInterface.with_fallback(fallback_func=exoplanet_fallback)
    def load_exoplanet_db(self, **kwargs) -> ApiResult:
        """
        Loads a file in pickle format.
        """
        file_path = kwargs.get("file_path")
        force_reload = kwargs.get("force_reload", False)
        reload_days = kwargs.get("reload_days", 7)
        if not os.path.is_file(file_path):
            raise FileNotFoundError(f"File not found: {file_path}, executing fallback")
        if force_reload:
            raise Exception(f"Force reload is enabled, executing fallback function")
        if os.path.getmtime(file_path) < time.time() - (reload_days * 24 * 60 * 60):
            raise Exception(f"File is older than {reload_days} days, executing fallback function")

        data = pd.read_pickle(file_path)
        return ApiResult(
            data=data,
            success=True,
            source=self.__class__.__name__
        )
    


class SimbadProvider(AstroqueryProvider):
    """
    Provider which handles queries to SIMBAD using ASTROQUERY
    """

    def __init__(self, verbose=False, simbad_votable_fields=None):
        self.simbad = aqs.Simbad()
        # CHECK VOTABLE FIELDS ON INIT


        if simbad_votable_fields:
            try:
                self.simbad.add_votable_fields(*simbad_votable_fields)
            except Exception as e:
                logger.error(f"Error adding votable fields: {e}")
                raise

    def support_params(self):
        """
        Common parameters for all astronomical queries using astroquery
        """
        return ["object_name", "verbose", "wildcard", "criteria", "get_query_payload"]

    @DataProviderInterface.with_fallback(return_empty_on_fail=True)
    def request(self, **kwargs) -> ApiResult:
        """
        Method that handles queries to SIMBAD using ASTROQUERY
        """
        format = kwargs.pop("format", "dict")
        object_name = kwargs.get("object_name")
        verbose = kwargs.pop("verbose", False) ## verbose cannot be pased to aqs since it is deprecated
        if not object_name:
            return ApiResult(
                data=None,
                success=False,
                error="Object name is required",
                source=self.__class__.__name__
            )
        if not isinstance(object_name, str):
            return ApiResult(
                data=None,
                success=False,
                error="Object name must be a string",
                source=self.__class__.__name__
            )


        # then we extract the parameters that are supported by the provider
        # and we pass them to the query
        kwargs = {key: value for key, value in kwargs.items() if key in self.support_params()}
        
        if verbose:
            logger.info(f"Querying SIMBAD: {object_name} with format {format} and params {kwargs}")
        # pass the extra arguments if needed
        response = self.simbad.query_object(**kwargs)
        if response is None:
            return ApiResult(
                data=None,
                success=False,
                error="No results found",
                source=self.__class__.__name__
            )
            
        else:

            return ApiResult(
                data=response,
                success=True,
                source=self.__class__.__name__
            )
        

    

class ExoplanetProvider(AstroqueryProvider): # USE THIS INSTEAD OF TAP
    """
    Provider which handles queries to the Nasa Exoplanet Archive using Astroquery
    """
    def __init__(self):
        self.exoplanet_archive = NasaExoplanetArchive()

    def support_params(self):
        return ["object_name", "verbose", "table", "get_query_payload", "regularize", "select", "where", "order"]
    
    @DataProviderInterface.with_fallback(return_empty_on_fail=True)
    def request(self, **kwargs) -> ApiResult:
        """
        Method that handles queries to the Nasa Exoplanet Archive using ASTROQUERY
        """
        # Extract the parameters that are supported by the provider
        kwargs = {key: value for key, value in kwargs.items() if key in self.support_params()}
        
        # Check if the object name is provided
        object_name = kwargs.get("object_name")
        if not object_name:
            return ApiResult(
                data=None,
                success=False,
                error="Object name is required",
                source=self.__class__.__name__
            )
        
        # Perform the query
        response = self.exoplanet_archive.query_object(**kwargs)
        
        if response is None:
            return ApiResult(
                data=None,
                success=False,
                error="No results found",
                source=self.__class__.__name__
            )
            
        else:
            return ApiResult(
                data=response,
                success=True,
                source=self.__class__.__name__
            )
    @DataProviderInterface.with_fallback(return_empty_on_fail=True)
    def query(self, **kwargs):
        """
        Method that handles queries to the Nasa Exoplanet Archive using ASTROQUERY
        
        Args:
            table: Table name to query (e.g., "ps" for Planetary Systems or "pscomppars")
            select: Comma-separated list of columns to return
            where: SQL WHERE clause to filter results
            order: Column(s) to order results by
        
        """


        kwargs = {key: value for key, value in kwargs.items() if key in self.support_params()}


        table = kwargs.get("table")  # Default to planetary systems
        select = kwargs.get("select")
        where = kwargs.get("where")
        order = kwargs.get("order")
        query_params = {}
        if table is not None:
            query_params["table"] = table
        if select is not None:
            query_params["select"] = select
        if where is not None:
            query_params["where"] = where
        if order is not None:
            query_params["order"] = order

        
        
        # Execute query
        response = self.exoplanet_archive.query_criteria(**query_params)
        
        if response is None or len(response) == 0:
            return ApiResult(
                data=None,
                success=False,
                error="No matching exoplanets found",
                source=self.__class__.__name__
            )
        
        return ApiResult(
            data=response,
            success=True,
            source=self.__class__.__name__
        )
        
class ApiService:
    """
    Main class used as interface to the API
    It will handle all the requests to the different providers.
    Args: 
        verbose: Whether to print the query. Optional
        simbad_votable_fields: List of votable fields to add to the SIMBAD provider. Optional
    """

    def __init__(self, verbose= False, simbad_votable_fields=None):
        self.simbad_provider = SimbadProvider()
        self.local_files_provider = LocalFilesProvider()
        self.local_files_provider.set_api_service(self)
        self.http_provider = HttpProvider()
        self.exoplanet_provider = ExoplanetProvider()
        self.verbose = verbose
        if simbad_votable_fields:
            self.add_simbad_votable_fields(*simbad_votable_fields)


    def get_provider(self, service):
        """
        Method that will return the provider for the given service.
        Args:
            service: Service to get the provider for. It can be "simbad", "http", "exoplanet", etc. 
        Returns:
            DataProviderInterface: Provider for the given service.
        """
        service = service.lower()
        if service == "simbad":
            return self.simbad_provider
        elif service == "http":
            return self.http_provider
        elif service == "exoplanet":
            return self.exoplanet_provider
        elif service == "localfiles":
            return self.local_files_provider
        # Add more providers as needed
        return None

    def request_http(self, **kwargs ) -> ApiResult:
        """
        Method that will handle HTTP requests using the HTTP provider.
        Args:
            url: URL to request. Mandatory
            method: HTTP method to use. Defaults to GET if not provided.
            headers: Headers to use for the request. Optional
            params: Parameters to use for the request. Optional
            data: Data to send with the request. Optional
            timeout: Timeout for the request. Optional
            verbose: Whether to print the query. Optional
            max_retries: Maximum number of retries for the request. Optional
        Returns:
            ApiResult: Result of the query. It will contain the data, success, error, source and is_fallback attributes.

        """
        kwargs["verbose"] = self.verbose
        return self.http_provider.request(**kwargs)


    def request_simbad(self, **kwargs) -> ApiResult:
        """
        Method that will handle queries to SIMBAD using the SIMBAD provider.
        Args:
            object_name: Name of the object to query. Mandatory
            verbose: Whether to print the query. Optional
            wildcard: Whether to use wildcard search. Optional
            criteria: Criteria to use for the query. Optional
            get_query_payload: Whether to get the query payload. Optional
        Returns:
            ApiResult: Result of the query. It will contain the data, success, error, source and is_fallback attributes.
        """
        kwargs["verbose"] = self.verbose
        return self.simbad_provider.request(**kwargs)

    def request_exoplanet(self, **kwargs) -> ApiResult:
        """
        Method that will handle queries to the Nasa Exoplanet Archive using Astroquery
        Args:
            object_name: Name of the object to query. Mandatory
            verbose: Whether to print the query. Optional
            table: Table to use for the query. Optional
            get_query_payload: Whether to get the query payload. Optional
            regularize : If True, the aliastable will be used to regularize the target name. Optional   
            **criteria: Any other filtering criteria to apply. Values provided using the where keyword will be ignored.


        Returns:
            ApiResult: Result of the query. It will contain the data, success, error, source and is_fallback attributes.
        """
        kwargs["verbose"] = self.verbose
        return self.exoplanet_provider.request(**kwargs)
    
    def add_simbad_votable_fields(self, *fields):
        """
        Method that will add votable fields to the SIMBAD provider.
        Args:
            fields: List of votable fields to add to the SIMBAD provider.
        """
        self.simbad_provider.simbad.add_votable_fields(*fields)



    def query_exoplanet(self, **kwargs):
        """
        Method that will handle queries to the Nasa Exoplanet Archive using Astroquery
        Args:
            table: Table name to query (e.g., "ps" for Planetary Systems or "pscomppars")
            select: Comma-separated list of columns to return
            where: SQL WHERE clause to filter results
            order: Column(s) to order results by
        Returns:
            ApiResult: Result of the query. It will contain the data, success, error, source and is_fallback attributes.

        Note:
            Astroquery automatically creates an extra column called sky_coord using ra and dec columns. 
        """
        kwargs["verbose"] = self.verbose
        return self.exoplanet_provider.query(**kwargs)
    

    def query_exoplanet_db(self, **kwargs):
        """
        Method that will seek a file called exodb.pickle. If the file does not exist, it will execute a fallback calling the Nasa Exoplanet Archive API
        Args:
            file_path: Path to the file to load. Mandatory
            force_reload: Whether to force reload the file using the fallback function (API call to NEA). Optional
            reload_days: Max age of the file to consider it old and execute the fallback function.
        Returns: 
            ApiResult object
        """
        return self.local_files_provider.load_exoplanet_db(**kwargs)
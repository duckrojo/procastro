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
from procastro import config
from procastro.api_provider.api_exceptions import (
    ApiServiceError, HttpProviderError, SimbadProviderError,
    ExoplanetProviderError, LocalFilesProviderError, DataValidationError
)

logger = logging.getLogger(__name__)
# type for results
T = TypeVar("T")  # this means that the result can be any type


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
                            fallback_data = fallback_func(self, e, **kwargs)
                            return ApiResult(
                                data=fallback_data,
                                success=True,
                                source=f"{self.__class__.__name__}_fallback",
                                is_fallback=True
                            )
                        except Exception as fallback_error:
                            raise ApiServiceError(
                                message=f"Fallback function failed: {fallback_error}",
                                details={"original_error": str(e)},
                                provider=self.__class__.__name__
                            )

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


class ApiResult(Generic[T]):
    """
    Generic container class, it will hold data of type T and T will be determined when the class is instantiated.
    Contains the result of a query to an API, including:
    - data: The actual data returned by the API.
    - success: Whether the request was successful or not.
    - error: Error message if the request failed.
    - source: Which provider returned this data.
    - is_fallback: Whether this data came from a fallback source (e.g., if the primary source failed).
    This class is used to provide a consistent interface for handling API responses, allowing for type hinting and better error handling.

    Example usage:
    ```python
        apiService = ApiService(verbose=True, simbad_votable_fields=["ra", "dec", "otype"])
        result = apiService.request_simbad(object_name="M31", wildcard=True)
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
        self.data = data  # the actual data returned
        self.success = success  # whether the request was successful or not
        self.error = error  # error message if the request failed
        self.source = source  # which provider returned this data
        self.is_fallback = is_fallback  # whether this came from a fallback source


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

    


class HttpProvider(DataProviderInterface):
    """
    Provider that will help to instantiate HTTP Providers . USGS and HORIZONS.
    """

    # NOTE: Here, we can set the set of parameters that this type of providers support.
    def support_params(self) -> list:
        """
        Method that will return a list of parameters that the provider supports.
        """
        return ["url", "headers", "params", "data", "timeout", "verbose", "method", "json", "cookies", "files", "auth", "allow_redirects", "proxies", "verify", "stream", "cert"]

    @with_fallback(return_empty_on_fail=True)
    def request(self,  **kwargs) -> ApiResult:
        """
        Method that handles HTTP requests
        """
        url = kwargs.get("url")
        method = kwargs.get("method", "GET")
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 1)
        # we pop verbose argument since it's not used by the request class.|
        verbose = kwargs.pop("verbose", False)

        if verbose:
            logger.info(f"HTTP request: {method} {url}")
        # NOTE: Here we use the request library to make the request.
        if not url:
            raise ApiServiceError(
                message="URL is required",
                details={"url": url},
                provider=self.__class__.__name__
            )

        kwargs = {key: value for key,
                  value in kwargs.items() if key in self.support_params()}
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

                if verbose:
                    logger.info(f"HTTP response: {response.status_code} ")
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
                logger.error(
                    f"HTTP error: {status_code} - for {url} - Error: {e}")

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
                    logger.error(
                        f"Request error: {url} - Error: {e}, retrying...")
                    # Exponential backoff
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                raise HttpProviderError(
                    message=f"Request failed after {max_retries} attempts: {url}",
                    url=url,
                )
            except requests.RequestException as e:
                logger.error(f"Request error: {url} - Error: {e}")
                raise HttpProviderError(
                    message=f"Request failed: {url}",
                    url=url
                )
            except requests.TooManyRedirects as e:
                logger.error(f"Too many redirects: {url} - Error: {e}")
                raise HttpProviderError(
                    message=f"Too many redirects: {url}",
                    url=url
                )


class AstroqueryProvider(DataProviderInterface):
    """
    Provider which handles astronomical queries using ASTROQUERY (like Nasa Exoplanet Archive, SIMBAD or others)

    """

    def __init__(self, simbad_votable_fields=None):
        self.simbad = aqs.Simbad()
        self.exoplanet_archive = NasaExoplanetArchive()
        if simbad_votable_fields:
            try:
                self.simbad.add_votable_fields(*simbad_votable_fields)
            except Exception as e:
                logger.error(f"Error adding votable fields: {e}")
                raise SimbadProviderError(
                    message=f"Error adding votable fields: {e}",
                    provider=self.__class__.__name__
                )

    def support_params(self, provider):
        """
        Common parameters for all astronomical queries using astroquery.
        We will be implementing this method on its child classes.
        """
        if provider == "simbad":
            return ["object_name", "verbose", "wildcard", "criteria", "get_query_payload"]
        elif provider == "exoplanet":
            return ["object_name", "verbose", "table", "get_query_payload", "regularize", "select", "where", "order"]

    @with_fallback(return_empty_on_fail=True)
    def request(self, provider: str, **kwargs) -> ApiResult:
        """
        Method that handles queries to Nasa Exoplanet Archive or SIMBAD depending on the {provider} argument.
        Args:
            provider: Provider to use for the query. It can be "simbad" or "exoplanet".
            **kwargs: Additional parameters to pass to the query.

        Returns:
            ApiResult: Result of the query. It will contain the data, success, error, source and is_fallback attributes.
        """
        provider: str = provider.lower()
        if provider not in ["simbad", "exoplanet"]:
            raise ApiServiceError(
                message=f"Provider {provider} is not supported",
                provider=self.__class__.__name__
            )
        # we pop verbose argument since it's not used by the request class.
        verbose = kwargs.pop("verbose", False)
        # Extract the parameters that are supported by the provider
        kwargs = {key: value for key, value in kwargs.items(
        ) if key in self.support_params(provider)}

        if verbose:
            logger.info(f"Querying {provider.upper()} with params: {kwargs}")

        provider = self.simbad if provider == "simbad" else self.exoplanet_archive

        try:
            response = provider.query_object(**kwargs)
            if response:
                return ApiResult(
                    data=response,
                    success=True,
                    source=self.__class__.__name__,
                )
        except Exception as e:
            exception_class = SimbadProviderError if provider == self.simbad else ExoplanetProviderError
            logger.error(f"Error querying SIMBAD: {e}")
            raise exception_class(
                message=f"Error querying SIMBAD: {e}",
            )

    @with_fallback(return_empty_on_fail=True)
    def query_nasa_exoplanet_archive(self, **kwargs):
        """
        Method that handles queries in SQL format to the Nasa Exoplanet Archive using ASTROQUERY. Replacement for the TAP service.

        Args:
            table: Table name to query (e.g., "ps" for Planetary Systems or "pscomppars")
            select: Comma-separated list of columns to return
            where: SQL WHERE clause to filter results
            order: Column(s) to order results by

        """

        kwargs = {key: value for key, value in kwargs.items(
        ) if key in self.support_params(provider="exoplanet")}

        table = kwargs.get("table")
        select = kwargs.get("select")
        where = kwargs.get("where")
        order = kwargs.get("order")
        # we pop verbose argument since it's not used by the request class.
        verbose = kwargs.pop("verbose", False)

        query_params = {}
        if table is not None:
            query_params["table"] = table
        if select is not None:
            query_params["select"] = select
        if where is not None:
            query_params["where"] = where
        if order is not None:
            query_params["order"] = order

        if not table:
            raise ExoplanetProviderError(
                message="Table name is required",
            )
        if not select:
            raise ExoplanetProviderError(
                message="Select clause is required",
            )
        if verbose:
            logger.info(f"Querying Exoplanet Archive: {query_params}")
        # Execute query
        try:
            response = self.exoplanet_archive.query_criteria(**query_params)
        except Exception as e:
            logger.error(f"Error querying Exoplanet Archive: {e}")
            raise ExoplanetProviderError(
                message=f"Error querying Exoplanet Archive: {e}",
            )

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


class LocalFilesProvider(DataProviderInterface):
    """
    Class to use when loading local files 
    """

    def __init__(self, api_service=None):
        self.api_service = api_service

    def support_params(self):
        return ["file_path", "force_reload", "reload_days"]

    def request(self):
        pass

    def fallback_transit(self, error, **kwargs):
        """
        Queries the Nasa Exoplanet Archive for the transit data of the target planet.
        Designed to be a fallback for the local files provider case.
        """
        target = kwargs.get("target")
        if not target:
            raise ApiServiceError(
                message="Target planet name is required for fallback query",
                details={"error": str(error)},
                provider=self.__class__.__name__
            )
        if self.api_service.verbose:
            logger.info(
                f"Querying Nasa Exoplanet Archive for transit data of {target}")
        response = self.api_service.query_exoplanet(
            table='pscomppars',
            select='pl_name, pl_tranmid, pl_orbper, pl_trandur',
            where=f"pl_name like '%{target}%'",
        )
        return response.data
    
    @with_fallback(fallback_func=fallback_transit)
    def load_transit_csv(self, file_path: str, target: str):
        """
        Loads a csv file and returns its content in the desired output_format.
        """
        if not os.path.exists(file_path):
            raise LocalFilesProviderError(
                message=f"File {file_path} does not exist, executing fallback query to NEA.",
                file_path=file_path
            )
        if self.api_service.verbose:
            logger.info(
                f"Attempting to load transit data from {file_path} for target {target}")

        # we receive a pandas dataframe.
        df = pd.read_csv(file_path)
        data = None
        for index, row in df.iterrows():
            planet_name = row['pl_name']
            if planet_name == target:
                transit_epoch = row['pl_tranmid']
                transit_period = row['pl_orbper']
                transit_length = row['pl_trandur']
                data = transit_epoch, transit_period, transit_length

        if data is not None:
            return ApiResult(
                data=data,
                success=True
            )
        else:
            raise LocalFilesProviderError(
                message=f"Target {target} not found in file {file_path}, executing fallback query to NEA",
            )




    @with_fallback(fallback_func=fallback_transit)
    def load_transit_txt_legacy(self, file_path: str, target: str):
        """
        Loads a txt file and returns its content in the desired output_format.
        """

        if not os.path.exists(file_path):
            raise LocalFilesProviderError(
                message=f"File {file_path} does not exist, executing fallback query to NEA.",
                file_path=file_path
            )
        if self.api_service.verbose:
            logger.info(
                f"Attempting to load transit data from {file_path} for target {target}")

        # we receive a list of strings .
        transits_list = open(file_path, "r").readlines()
        data = None
        for transit in transits_list:
            # first, the elements are in the format:
            # planet_name E[transit_epoch] P[transit_period] L[transit_length]
            # the values are separated by spaces Planet names have to be saved with and undescore replacing spaces  .
            # some of the values can be ommited, so we have to parse them using the first letter of the value,
            # E is transit epoch, P is the transit period and L the transit length
            values = transit.split()
            planet_name = values[0]
            if planet_name == target:
                transit_epoch = transit_period = transit_length = None
                for value in values[1:]:
                    if value.startswith("E"):
                        transit_epoch = value[1:-1]
                    elif value.startswith("P"):
                        transit_period = value[1:-1]
                    elif value.startswith("L"):
                        transit_length = value[1:-1]
                data = transit_epoch, transit_period, transit_length

        if data != None:
            return ApiResult(
                data=data,
                success=True
            )
        else:
            raise LocalFilesProviderError(
                message=f"Target {target} not found in file {file_path}, executing fallback query to NEA",
            )


class ApiService:
    """
    Main class used as interface to the API
    It will handle all the requests to the different providers.
    Args: 
        verbose: Whether to print the query. Optional
        simbad_votable_fields: List of votable fields to add to the SIMBAD provider. Optional
    Usage:
        api_service = ApiService(verbose=True, simbad_votable_fields=["ra", "dec", "otype"])
        response = api_service.request_simbad(object_name="M31", wildcard=True)
        if response.success:
            print(f"Data for M31: {response.data}")
        else:
            print(f"Error querying SIMBAD: {response.error}")

    """

    def __init__(self, verbose=False, simbad_votable_fields=None):

        self.local_files_provider = LocalFilesProvider(api_service=self)
        self.http_provider = HttpProvider()
        self.astroquery_provider = AstroqueryProvider(
            simbad_votable_fields=simbad_votable_fields)
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
        elif service == "astroquery":
            return self.astroquery_provider
        elif service == "localfiles":
            return self.local_files_provider
        # Add more providers as needed
        return None

    def request_http(self, **kwargs) -> ApiResult:
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
        return self.astroquery_provider.request(provider="simbad", **kwargs)

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
        return self.astroquery_provider.request(provider="exoplanet", **kwargs)

    def add_simbad_votable_fields(self, *fields):
        """
        Method that will add votable fields to the SIMBAD provider.
        Args:
            fields: List of votable fields to add to the SIMBAD provider.
        """
        self.simbad_provider.simbad.add_votable_fields(*fields)

    def query_exoplanet(self, **kwargs):
        """
        Method that will handle queries directly to the Nasa Exoplanet Archive using Astroquery
        Args:
            table: Table name to query (e.g., "ps" for Planetary Systems or "pscomppars")
            select: Comma-separated list of columns to return
            where: SQL WHERE clause to filter results
            order: Column(s) to order results by
        Returns:
            ApiResult: Result of the query. It will contain the data, success, error, source and is_fallback attributes.


        Example usage:
        ```python
        target = "Kepler-22b"
        api_service = ApiService(verbose=True)
        response: ApiResult = api_service.query_exoplanet(
                table = 'pscomppars',
                selection='pl_name, pl_tranmid, pl_orbper, pl_trandur',
                where = f"lower(pl_name) like '%{target}%'",
            )
        if response.success:
            print(f"Exoplanets found: {response.data}")
        else:
            print(f"Error querying Exoplanet Archive: {response.error}")
        ```
        Note:
            Astroquery automatically creates an extra column called sky_coord using ra and dec columns. 
        """
        kwargs["verbose"] = self.verbose
        return self.astroquery_provider.query_nasa_exoplanet_archive(**kwargs)

    def query_transits_ephemeris(self, file_path: str, target: str, update: bool = False):
        """
        Method that will handle queries to the local files provider and if the method fails, it will try to query the Nasa Exoplanet Archive.
        Args:
            file_path: Path of the local file. In this case, it defaults to transits.txt
            update: If True, it will update the local file with the latest data from NEA overwriting the target planet data.
            target: Name of the target planet.  
        Returns:
            Transit epoch, Transit Period, Transit Length. 
        """

        file_type = file_path.split(".")[-1].lower()
        if file_type not in ["txt", "csv"]:
            raise ApiServiceError(
                message=f"File type {file_type} is not supported. Supported types are: txt, csv",
                details={"file_type": file_type},
                provider=self.__class__.__name__
            )
        if self.verbose:
            logger.info(f"Querying transits ephemeris for {target} in {file_path} with file type {file_type}")
        
        if not target:
            raise ApiServiceError(message="Target planet name is required")
        

        
        
        target_with_underscore = target.replace(" ", "_")
        
        # Get response based on file type
        response = (self.local_files_provider.load_transit_txt_legacy(file_path=file_path, target=target_with_underscore) 
                    if file_type == "txt" 
                    else self.local_files_provider.load_transit_csv(file_path=file_path, target=target_with_underscore))
        
        if not response.success:
            logger.error(f"Error loading transit data for {target} from {file_path}: {response.error}")
            return None
        
        # Log source of data
        if self.verbose:
            source = "Nasa Exoplanet Archive" if response.is_fallback else file_path
            logger.info(f"Loaded transit data for {target} from {source}")
        
        # Extract transit data
        if response.is_fallback:
            df = response.data.to_pandas()
            pl_tranmid, pl_orbper, pl_trandur = df.iloc[0][['pl_tranmid', 'pl_orbper', 'pl_trandur']]
            
            if update:
                self._update_transit_file(file_path, file_type, target_with_underscore, 
                                        pl_tranmid, pl_orbper, pl_trandur)
        else:
            pl_tranmid, pl_orbper, pl_trandur = response.data
        
        return pl_tranmid, pl_orbper, pl_trandur
    
    def _update_transit_file(self, file_path: str, file_type: str, target: str, 
                            pl_tranmid, pl_orbper, pl_trandur):
        """Helper method to update transit files with new data"""
        if file_type == "txt":
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            # Find and update existing line or append new one
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{target} "):
                    lines[i] = f"{target} E{pl_tranmid} P{pl_orbper} L{pl_trandur} CNasaExoplanetArchiveSource\n"
                    updated = True
                    break
            
            if not updated:
                lines.append(f"{target} E{pl_tranmid} P{pl_orbper} L{pl_trandur} CNasaExoplanetArchiveSource\n")
            
            with open(file_path, "w") as f:
                f.writelines(lines)
        else:  # csv
            df = pd.read_csv(file_path)
            new_data = {'pl_name': target, 'pl_tranmid': pl_tranmid, 'pl_orbper': pl_orbper, 
                       'pl_trandur': pl_trandur, 'source': "NasaExoplanetArchiveSource"}
            
            if target in df['pl_name'].values:
                df.loc[df['pl_name'] == target, ['pl_tranmid', 'pl_orbper', 'pl_trandur']] = pl_tranmid, pl_orbper, pl_trandur
            else:
                new_row = pd.DataFrame([new_data])
                df = pd.concat([df, new_row], ignore_index=True)
            
            df.to_csv(file_path, index=False)
        
        if self.verbose:
            logger.info(f"Updated transit data for {target} in {file_path}")
                
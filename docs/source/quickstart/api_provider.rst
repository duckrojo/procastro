Api Provider
================

ProcAstro handles data requests using a variety of sources, varying from SIMBAD, ASTROQUERY, USGS and local files

ApiResult 
----------------
The ApiResult class is a container for the results of an API call. 
It includes ...

.. code-block:: python

    class ApiResult:
        def __init__(self, data=None, success=True, error=None, source=None, is_fallback=False):
            """
            Initialize an ApiResult instance.

            :param data: The actual data returned from the API call.
            :param success: Whether the request was successful or not.
            :param error: Error message if the request failed.
            :param source: Which provider returned this data.
            :param is_fallback: Whether this came from a fallback source.
            """

Every Api Result must come with the desired format.

Fallback executioner
--------------------------------
ProcAstro uses a fallback executioner to handle exceptions or errors during the data retrieval.
It works by using a decorator that wraps the API call function.
If an error occurs, it will try to execute the fallback function.


Code example: 

.. code-block:: python

    from procastro.api_provider import ApiResult, with_fallback


    def fallback_function(**kwargs):
        """
        Fallback function to be executed in case of an error.
        This function can be used to provide alternative data or handle errors gracefully.
        """
        return ApiResult(data='data', success=False, error="Fallback executed", source="fallback")

    @with_fallback(fallback_function)
    def api_call_function(**kwargs):
        """
        Main API call function that retrieves data from the primary source.
        If an error occurs, the fallback function will be executed.
        """
        # Simulate an API call that might fail
        raise Exception("Simulated API error")

In this case, the original function `api_call_function` will raise an exception, and the fallback function will be executed instead.
The result will be an `ApiResult` object with the data from the fallback function.


The possible params of `with_fallback` decorator are:

- `fallback_func`: The fallback function to be executed in case of an error.
- `return_empty_on_fail`: If set to `True`, the original function will not raise an exception, but will return an empty result if it fails. If set to `False`, it will raise an exception if the original function fails.


The decorator can be used to wrap any function that retrieves data from an API, allowing for graceful error handling and fallback mechanisms.

This is an example of an actual API call using the decorator:


.. code-block:: python

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

We can clearly see the pipeline of this request. 

* First, it tries to load the file from the local files provider.
* If the file does not exist, it raises an exception and executes the fallback function.
* The fallback function queries the Nasa Exoplanet Archive for the transit data of the target planet.
* If the target planet is found, it returns the data in the desired format.
* If the target planet is not found, it raises an exception.


DataProviderInterface class
----------------

The `DataProviderInterface` class is the main class that handles the API calls and provides the data to the user.

It is organized in the following way:

- The DataProviderInterface Class, which is the base class for all data providers.
    - It defines the methods that must be implemented by all data providers. (In this case, request and support params)

The following classes inherit from the `DataProviderInterface` class:

- HttpProvider: 
    - Handles HTTP requests to external APIs.
    - It can be used to query data from any API that supports HTTP requests.
    - Currently, USGS, HORIZONS use this provider.

    .. code-block:: python

        from procastro.api_provider import ApiService

        api = ApiService(verbose=True)
        result = api.request_http(url, method)
        data = result.data

- AstroqueryProvider
    - Handles queries to the Astroquery library. 
    - Currently, SIMBAD and the NASA EXOPLANET ARCHIVE use this provider.
    - It can be used to query data from any service that is supported by Astroquery.
    - It depends on the `astroquery` library, which must be installed in your environment.    
    Since astroquery supports query-like requests, AstroqueryProvider has a `query_nasa_exoplanet_archive` method that can be used to query data from the service.
    
    
    .. code-block:: python

        from procastro.api_provider import ApiService

        apiService = ApiService(verbose=True)
            resultset_new: ApiResult = apiService.query_exoplanet(
            table= "pscomppars",
            select = "pl_name,ra,dec,pl_orbper,pl_tranmid,disc_facility,pl_trandur,sy_pmra,sy_pmdec,sy_vmag",
            where = "pl_tranmid!=0.0 and pl_orbper!=0.0",
        )
        data = result.data

    And object focused requests ...

    .. code-block:: python

        apiService = ApiService(verbose= True)
        response = apiService.request_simbad(object_name= "M [1-9]", wildcard=True)
        
    .. 


- LocalFilesProvider
    - Handles requests to local files.
    - Currently supports loading transit files from local or NasaExoplanetArchive
    .. code-block:: python

        from procastro.api_provider import ApiService
        target = "WASP-5 b"
        paths_transits = [config_exo['transit_file'],
                    ]
        response = ApiService(verbose=True).query_transits_ephemeris(
        file_path=paths_transits[0],
        target=target,
        update=False,
        )

    The update parameter states when the NEA queries will update the local file if the found the specified target.


The ApiService Interface
----------------

The `ApiService` class is the main entry point for the API provider. It handles the requests and provides the data to the user.
Below is a diagram of the ApiService class and its dependencies. 


.. image :: ../figs/api_provider_diagram.png
   :width: 700
   :alt: ApiService class diagram


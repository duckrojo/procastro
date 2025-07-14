from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests
import pyvo as vo
import pyvo.dal.exceptions
from astropy.table import QTable
import procastro as pa
from procastro.api_provider.api_exceptions import ApiServiceError, HttpProviderError
from procastro.api_provider.api_service import ApiResult, ApiService, DataProviderInterface, HttpProvider, with_fallback 

@pytest.fixture
def apiService():
    return ApiService()



#TODO: TEST WITH NEW ASTROQUERY PROVIDER
################ API RESULT TESTS #######################
def test_api_result_initialization():
    result = ApiResult(
        data = "data",
        success = True,
    )
    assert result.data == "data"
    assert result.success is True
    assert result.error is None
    assert result.is_fallback is False



################ FALLBACK TESTS #######################
def test_fallback():
    
    class MockedProvider(DataProviderInterface):

        def support_params(self):
            return ['foo', 'bar']


        @with_fallback(fallback_func=lambda *args, **kwargs: "fallback executed")
        def request(self):
            raise Exception("error")

    provider = MockedProvider()
    result = provider.request()
    assert result.success is True
    assert result.data == "fallback executed"
    assert result.is_fallback is True
    assert result.source == MockedProvider.__name__ + "_fallback"

def test_empty_on_fail_with_no_fallback_provided():

    class MockedProvider(DataProviderInterface):

        def support_params(self):
            return ['foo', 'bar']


        @with_fallback(fallback_func=None, return_empty_on_fail=True)
        def request(self):
            raise Exception("error")
        


    provider = MockedProvider()
    result = provider.request()
    assert result.success is False
    assert result.data is None
    assert result.error is not "error"
    assert result.is_fallback is False
    assert result.source == MockedProvider.__name__ 


def test_fallback_with_fallback_error():
    def failing_fallback(*args, **kwargs):
        raise Exception('error')

    class MockedProvider(DataProviderInterface):
        def support_params(self):
            return ['foo', 'bar']

        @with_fallback(fallback_func=failing_fallback)
        def request(self):
            raise Exception("error")
            
    provider = MockedProvider()    
    with pytest.raises(ApiServiceError) as excinfo:
        provider.request()

    # Check that the error message contains the required parts
    assert "Fallback function failed: error" in str(excinfo.value)
    assert "Provider: MockedProvider" in str(excinfo.value)
    
    # If you want to check the details attribute specifically
    if hasattr(excinfo.value, 'details'):
        assert excinfo.value.details.get('original_error') == 'error'

############### HTTTP PROVIDER TESTS #######################

def test_http_provider_request():

    #### test with a not valid URL ####
    provider = HttpProvider()
    result = provider.request()
    assert result.success is False
    assert result.data is None
    assert isinstance(result.error, ApiServiceError)
    assert "URL is required" in str(result.error)
    assert result.source == HttpProvider.__name__


    #### test with a non valid parameter (provider must ignore it)####
    result1 = provider.request(url="https://example.com", invalid_param= "foo")
    assert result1.success is True
    assert result1.data is not None
    assert result1.error is None
    assert result1.source == HttpProvider.__name__
    assert result1.is_fallback is False


    #### tests with a suite of different status codes ####
    base_url = "https://httpstat.us/" # page that returns the status code in the body of the response
    status_codes = [200, 400, 401, 403, 404, 500]
    expected_results = {
        200: True,
        400: False,
        401: False,
        403: False,
        404: False,
        500: False
    }
    for status_code in status_codes:
        url = f"{base_url}{status_code}"
        result = provider.request(url=url)
        assert result.success == expected_results[status_code]
        if status_code == 200:
            assert result.data is not None
            assert result.error is None
            assert result.source == HttpProvider.__name__
            assert result.is_fallback is False
        else:
            assert result.data is None
            assert result.error is not None
            assert result.source == HttpProvider.__name__
            assert result.is_fallback is False


def test_http_post_request():
    apiService = ApiService(verbose = True)
    url = "https://httpbin.org/post"
    data = {"key": "value"}
    result = apiService.request_http(url=url, data=data, method="POST")
    assert result.success is True
    assert result.data is not None
    assert result.error is None
    assert result.source == HttpProvider.__name__
    assert result.is_fallback is False

def tests_http_timeout():
    provider = HttpProvider()
    max_retries = 1  # Number of retries
    result = provider.request(
        url="https://httpbin.org/delay/10",  # URL that takes 10 seconds to respond
        timeout=0.1,  # Very short timeout
        max_retries=max_retries
    )
    # HTTP provider is configured to NOT raise an exception on timeout (return_empty_on_fail=True)
    # So we expect a failed result with success=False and error message indicating timeout
    print(result.error)
    assert result.success is False
    assert result.data is None
    
    # Check for key components instead of exact match
    error_str = str(result.error)
    assert isinstance(result.error, HttpProviderError)
    assert "Request failed" in error_str
    assert "httpbin.org/delay/10" in error_str
    assert "Provider: HttpProvider" in error_str

def test_http_request_exception():
    with patch('requests.request') as mock_request:
        # Simulate a generic RequestException
        mock_request.side_effect = requests.RequestException("Simulated request exception")
        
        provider = HttpProvider()
        result = provider.request(url="https://example.com", max_retries=1)
        
        # Test with fallback behavior
        assert result.success is False
        assert result.data is None
        assert isinstance(result.error, HttpProviderError)
        assert "Request failed" in str(result.error)
        assert "https://example.com" in str(result.error)

#### TESTS SIMBAD PROVIDER ####
def test_simbad_provider():
    apiService = ApiService(verbose= True)
    response  = apiService.request_simbad(object_name= "m1")
    assert response.success is True
    assert response.data is not None
    assert response.error is None
    assert response.source == "AstroqueryProvider"
    assert response.is_fallback is False



def test_simbad_provider_with_extra_args():
    ## test with more args
    apiService = ApiService(verbose= True)
    response = apiService.request_simbad(object_name= "m1", dumb_arg = "foo", another_arg= "bar")
    assert response.success is True
    assert response.data is not None
    assert response.error is None
    assert response.source == "AstroqueryProvider"
    assert response.is_fallback is False


def test_simbad_provider_with_valid_args():
    ## test with valid args
    apiService = ApiService(verbose= True)
    response = apiService.request_simbad(object_name= "M [1-9]", wildcard=True)
    assert response.success is True
    assert response.data is not None
    assert response.error is None
    assert response.source == "AstroqueryProvider"
    assert response.is_fallback is False


def test_simbad_with_bad_args():
    ## test with bad args
    apiService = ApiService(verbose=True)
    response = apiService.request_simbad(object_name=1)
    assert response.success is False
    assert response.data is None
    
    # Check for the correct error type and message
    from procastro.api_provider.api_exceptions import AstroqueryProviderError
    assert isinstance(response.error,AstroqueryProviderError)
    assert response.source == "AstroqueryProvider"
    assert response.is_fallback is False


#### TEST EXOPLANET PROVIDER ####
def test_exoplanet_provider():
    apiService = ApiService(verbose= True)
    response = apiService.request_exoplanet(object_name= "k2-18 b")
    assert response.success is True
    assert response.data is not None
    assert response.error is None
    assert response.source == "AstroqueryProvider"
    assert response.is_fallback is False


def test_exoplanet_provider_query():
    
    exo_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    resultset_old = exo_service.search(
        f"SELECT pl_name,ra,dec,pl_orbper,pl_tranmid,disc_facility,pl_trandur,sy_pmra,sy_pmdec,sy_vmag,sy_gmag "
        f"FROM exo_tap.pscomppars "
        f"WHERE pl_tranmid!=0.0 and pl_orbper!=0.0 ")
    

    apiService = ApiService(verbose = True)
    resultset_new: ApiResult = apiService.query_exoplanet(
        table= "pscomppars",
        select = "pl_name,ra,dec,pl_orbper,pl_tranmid,disc_facility,pl_trandur,sy_pmra,sy_pmdec,sy_vmag",
        where = "pl_tranmid!=0.0 and pl_orbper!=0.0",
    )
    assert resultset_new.success is True
    assert resultset_new.data is not None
    assert resultset_new.error is None
    assert resultset_new.source == "AstroqueryProvider"
    assert resultset_new.is_fallback is False
    df_new = resultset_new.data.to_pandas()
    df_old = resultset_old.to_table().to_pandas()
    # as said in the apiService, the query method query_criteria returns an extra column called "sky_coord", so , for testing purposes, we will compare the intersection
    # between the two dataframes.
    common_cols = set(df_new.columns) & set(df_old.columns)
    df_new_common = df_new[list(common_cols)]
    df_old_common = df_old[list(common_cols)]
    pd.testing.assert_frame_equal(
        df_new_common.reset_index(drop=True).sort_index(axis=1),
        df_old_common.reset_index(drop=True).sort_index(axis=1),
        check_dtype=False,
        check_like=True
    )


def test_local_files_provider():
    from procastro import config
    config_exo = config.config_user("exoplanet")
    target = "WASP-5 b"
    paths_transits = [config_exo['transit_file'],
                    ]
    
    file_type = paths_transits[0].split('.')[-1]
    if file_type == "csv":
        df = pd.read_csv(paths_transits[0])
        # calculate the number of rows of the dataframe.
        n_rows = len(df)

    elif file_type == "txt":
        # load the txt file and count the number of lines
        with open(paths_transits[0], 'r') as f:
            lines = f.readlines()
            n_rows = len(lines)
    response = ApiService(verbose=True).query_transits_ephemeris(
        file_path=paths_transits[0],
        target=target,
        update=False,
    )
    assert response is not None
    assert type(response) is tuple
    assert len(response) == 3

    # as update is set to false, we expect the data to be the same as the one in the file
    if file_type == "csv":
        df_new = pd.read_csv(paths_transits[0])
        n_new = len(df_new)
        assert n_new == n_rows
    elif file_type == "txt":
        with open(paths_transits[0], 'r') as f:
            lines_new = f.readlines()
            n_new = len(lines_new)
        assert n_new == n_rows


    # now with update= True
    response = ApiService(verbose=True).query_transits_ephemeris(
        file_path=paths_transits[0],
        target=target,
        update=True,
    )






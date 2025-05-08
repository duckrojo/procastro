from unittest.mock import MagicMock, patch
import pytest
import requests

from procastro.api_provider.api_service import ApiResult, ApiService, DataProviderInterface, HttpProvider 

@pytest.fixture
def apiService():
    return ApiService()


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


        @DataProviderInterface.with_fallback(fallback_func=lambda *args, **kwargs: "fallback executed")
        def get_data(self):
            raise Exception("error")

    provider = MockedProvider()
    result = provider.get_data()
    assert result.success is True
    assert result.data == "fallback executed"
    assert result.is_fallback is True
    assert result.source == MockedProvider.__name__ + "_fallback"

def test_empty_on_fail_with_no_fallback_provided():

    class MockedProvider(DataProviderInterface):

        def support_params(self):
            return ['foo', 'bar']


        @DataProviderInterface.with_fallback(fallback_func=None, return_empty_on_fail=True)
        def get_data(self):
            raise Exception("error")
        


    provider = MockedProvider()
    result = provider.get_data()
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


        @DataProviderInterface.with_fallback(fallback_func=failing_fallback)
        def get_data(self):
            raise Exception("error")
    provider = MockedProvider()    
    with pytest.raises(Exception) as excinfo:
        
        provider.get_data()


    assert str(excinfo.value) == "Fallback function failed: error"


############### HTTTP PROVIDER TESTS #######################

def test_http_provider_supported_params():

    provider = HttpProvider()
    assert provider.support_params() == ["url", "headers", "params", "data", "timeout", "verbose"]

def test_http_provider_get_data():

    #### test with a not valid URL ####
    provider= HttpProvider()
    result = provider.get_data()
    assert result.success is False
    assert result.data is None
    assert result.error == "URL is required"
    assert result.source == HttpProvider.__name__


    #### test with a non valid parameter (provider must ignore it)####
    result1 = provider.get_data(url="https://example.com", invalid_param= "foo")
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
        result = provider.get_data(url=url)
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




def tests_http_timeout():
    provider = HttpProvider()
    max_retries = 1  # Number of retries
    result = provider.get_data(
        url="https://httpbin.org/delay/10",  # Esta URL tarda 10 segundos en responder
        timeout=0.1,  # Timeout muy corto (0.1 segundos)
        max_retries=max_retries  # Solo un intento
    )
    # HTTP provider is configured to NOT raise an exception on timeout, because return_empty_on_fail is set to True
    # So we expect a failed result with success=False and error message indicating timeout
    print(result.error)
    assert result.success is False
    assert result.data is None
    assert str(result.error) == f"Request failed after {max_retries} retries: Timeout"
    
def test_http_request_exception():
    with patch('requests.request') as mock_request:
        # Simular una RequestException genérica
        mock_request.side_effect = requests.RequestException("Simulated request exception")
        
        provider = HttpProvider()
        result = provider.get_data(url="https://example.com", max_retries=1)
        
        # Con return_empty_on_fail=True, debemos obtener un ApiResult con error
        assert result.success is False
        assert result.data is None
        assert "Request failed: Simulated request exception" in str(result.error)
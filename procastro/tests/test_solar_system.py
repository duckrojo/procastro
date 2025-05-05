import pytest
import numpy as np
from astropy.table import Table
import io
import sys
from contextlib import redirect_stdout
from procastro.astro.solar_system import BodyGeometry

@pytest.fixture
def mock_ephemeris():
    """Crear datos de efemérides simulados para pruebas."""
    data = {
        'ObsSub_LON': 120.0, #TODO: El ángulo de emision debe de ser 90º
        'ObsSub_LAT': 30.0, # TODO. Verificar el lado opuesto
        'NP_ang': 15.0,
        'SunSub_LON': 90.0, #TODO: Debo recibir la hora local del medio dia  (probnar)
        'SunSub_LAT': 10.0, 
        'SN_ang': 45.0,
        'SN_dist': 35.5,
        'NP_dist': 25.0,  # Añadir este campo faltante
        'Ang_diam': 30.0,  # 30 arcsec
        'jd': 2459000.5
    }
    #TODO : Verificar el lado opuesto (cambiar el signo de la latitud y sumarle 180 a la longitud)
    return Table([data])[0]  # Devuelve una fila de Tabla

class TestBodyGeometry:
    
    def test_init(self, mock_ephemeris):
        """Probar la correcta inicialización con datos de efemérides."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Verificar que los atributos básicos se asignan correctamente
        assert geometry.sub_obs == (120.0, 30.0)
        assert geometry.sub_obs_np == 15.0
        assert geometry.sub_sun == (90.0, 10.0)
        assert geometry.ang_diam == 30.0
        
        # Verificar que se crean los objetos de rotación
        assert geometry._rotate_to_subobs is not None
        assert geometry._rotate_to_subsol is not None
    
    def test_print(self, mock_ephemeris):
        """Probar que el método print muestra la información correctamente."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Capturar la salida estándar para verificar
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            geometry.print()
        
        output = captured_output.getvalue()
        
        # Verificar que la salida contiene la información clave
        assert "Sub Observer longitude/latitude: 120.0/30.0" in output
        assert "Sub Solar longitude/latitude: 90.0/10.0" in output
        assert "Sub Solar angle/distances w/r to sub-observer: 45.0/35.5" in output
        assert "North Pole angle/distances w/r to sub-observer: 15.0" in output
    
    def test_location_visible(self, mock_ephemeris):
        """Probar el cálculo de ubicación para un punto visible."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Ubicación en la cara visible (cerca de sub_obs)
        result = geometry.location(115.0, 25.0)
        
        # Verificar que se reporta como visible
        assert result['visible'] == True
        
        # Verificar valores clave
        assert isinstance(result['delta_ra'], float)
        assert isinstance(result['delta_dec'], float)
        assert 0 <= result['local_time'] <= 24  # Tiempo local válido
        assert 0 <= result['incoming'] <= 180   # Ángulo de incidencia válido
        assert 0 <= result['outgoing'] <= 90    # Ángulo de emisión para punto visible
    
    def test_location_invisible(self, mock_ephemeris):
        """Probar el cálculo de ubicación para un punto invisible."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Ubicación en la cara oculta (opuesta a sub_obs)
        result = geometry.location(300.0, -30.0)  # Opuesta a (120, 30)
        
        # Verificar que se reporta como no visible
        assert result['visible'] == False
    
    def test_location_with_label(self, mock_ephemeris):
        """Probar la opción de etiqueta para mostrar información."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Capturar la salida para verificar el formato
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            geometry.location(115.0, 25.0, label="TestCrater", max_title_len=10)
        
        output = captured_output.getvalue()
        
        # Verificar que la etiqueta aparece correctamente
        assert "TestCrater" in output
        assert "LocalSolarTime" in output
        assert "inc/emi ang" in output
    
    def test_location_poles(self, mock_ephemeris):
        """Probar el cálculo para ubicaciones en los polos."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Polos Norte y Sur
        north_pole = geometry.location(0, 90)
        south_pole = geometry.location(0, -90)
        
        # Al rotar en longitud, los polos deberían mantener las mismas coordenadas
        north_pole_rotated = geometry.location(120, 90)
        assert abs(north_pole['delta_ra'] - north_pole_rotated['delta_ra']) < 1e-10
        assert abs(north_pole['delta_dec'] - north_pole_rotated['delta_dec']) < 1e-10
        
        # Los polos deben tener posiciones opuestas en delta_dec (si ambos son visibles)
        if north_pole['visible'] and south_pole['visible']:
            assert (north_pole['delta_dec'] * south_pole['delta_dec']) < 0
        
    def test_day_night_terminator(self, mock_ephemeris):
        """Probar ubicaciones cerca del terminador día/noche."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Buscar un punto en el terminador día/noche
        # La longitud perpendicular al sub solar tendrá puntos en el terminador
        perpendicular_lon = (geometry.sub_sun[0] + 90) % 360
        
        # Probar varios puntos con esta longitud
        for lat in range(-80, 81, 20):
            result = geometry.location(perpendicular_lon, lat)
            # En el terminador, el ángulo de incidencia solar debería estar cerca de 90°
            assert 80 <= result['incoming'] <= 100
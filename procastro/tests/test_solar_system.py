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
    def test_location_at_specific_points(self, mock_ephemeris):
        """Test location calculation at specific points with known expected results."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Test at sub-observer point
        sub_obs_result = geometry.location(mock_ephemeris['ObsSub_LON'], mock_ephemeris['ObsSub_LAT'])
        assert sub_obs_result['visible'] == True
        assert abs(sub_obs_result['outgoing']) < 0.1  # Emission angle should be close to 0
        assert abs(sub_obs_result['delta_ra']) < 0.1  # Should be at center
        assert abs(sub_obs_result['delta_dec']) < 0.1  # Should be at center
        
        # Test at sub-solar point
        sub_sun_result = geometry.location(mock_ephemeris['SunSub_LON'], mock_ephemeris['SunSub_LAT'])
        assert sub_sun_result['visible'] == True  # The sub-solar point should be visible
        assert abs(sub_sun_result['incoming']) < 0.1  # Incidence angle should be close to 0
        assert abs(sub_sun_result['local_time'] - 12.0) < 0.1  # Local time should be noon
        
        # Test at anti-observer point (opposite to sub-observer)
        anti_obs_lon = (mock_ephemeris['ObsSub_LON'] + 180) % 360
        anti_obs_lat = -mock_ephemeris['ObsSub_LAT']
        anti_obs_result = geometry.location(anti_obs_lon, anti_obs_lat)
        assert anti_obs_result['visible'] == False  # Should be invisible

    def test_local_time_across_longitudes(self, mock_ephemeris):
        """Test that local time varies correctly across longitudes."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # At the sub-solar longitude, local time should be 12:00
        noon_result = geometry.location(mock_ephemeris['SunSub_LON'], 0)
        assert abs(noon_result['local_time'] - 12.0) < 0.1
        
        # Test at different longitudes around the body
        test_points = [
            ((mock_ephemeris['SunSub_LON'] + 90) % 360, 18.0),  # 90° east: 6pm
            ((mock_ephemeris['SunSub_LON'] + 180) % 360, 0.0),  # 180° east: midnight
            ((mock_ephemeris['SunSub_LON'] - 90) % 360, 6.0),   # 90° west: 6am
        ]
        
        for lon, expected_time in test_points:
            result = geometry.location(lon, 0)
            # Allow for 24-hour wrapping
            local_time = result['local_time']
            if local_time > 12 and expected_time < 12:
                local_time -= 24
            elif local_time < 12 and expected_time > 12:
                local_time += 24
            assert abs(local_time - expected_time) < 0.1

    def test_angular_diameter_effect(self, mock_ephemeris):
        """Test that angular diameter correctly affects the projected offsets."""
        # Create two geometries with different angular diameters
        small_data = {k: mock_ephemeris[k] for k in mock_ephemeris.colnames}
        small_data['Ang_diam'] = 15.0  # Half the original
        small_ephemeris = Table([small_data])[0]
        
        large_data = {k: mock_ephemeris[k] for k in mock_ephemeris.colnames}
        large_data['Ang_diam'] = 60.0  # Double the original
        large_ephemeris = Table([large_data])[0]
        
        small_geometry = BodyGeometry(small_ephemeris)
        original_geometry = BodyGeometry(mock_ephemeris)
        large_geometry = BodyGeometry(large_ephemeris)
        
        # Test at a consistent location
        test_lon, test_lat = 110.0, 20.0  # A visible point
        
        small_result = small_geometry.location(test_lon, test_lat)
        original_result = original_geometry.location(test_lon, test_lat)
        large_result = large_geometry.location(test_lon, test_lat)
        
        # Check that deltas scale proportionally with angular diameter
        assert abs(small_result['delta_ra'] / original_result['delta_ra'] - 0.5) < 0.01
        assert abs(small_result['delta_dec'] / original_result['delta_dec'] - 0.5) < 0.01
        
        assert abs(large_result['delta_ra'] / original_result['delta_ra'] - 2.0) < 0.01
        assert abs(large_result['delta_dec'] / original_result['delta_dec'] - 2.0) < 0.01
        
        # But normalized unit coordinates should remain the same
        assert abs(small_result['unit_ra'] - original_result['unit_ra']) < 0.01
        assert abs(small_result['unit_dec'] - original_result['unit_dec']) < 0.01

    def test_north_pole_effect(self, mock_ephemeris):
        """Test how the North Pole angle affects the projected coordinates."""
        # Create two geometries with different North Pole angles
        normal_data = {k: mock_ephemeris[k] for k in mock_ephemeris.colnames}
        normal_ephemeris = Table([normal_data])[0]
        
        rotated_data = {k: mock_ephemeris[k] for k in mock_ephemeris.colnames}
        rotated_data['NP_ang'] = mock_ephemeris['NP_ang'] + 90.0  # Rotate 90°
        rotated_ephemeris = Table([rotated_data])[0]
        
        normal_geometry = BodyGeometry(normal_ephemeris)
        rotated_geometry = BodyGeometry(rotated_ephemeris)
        
        # Test at a point near the pole
        test_lon, test_lat = 0, 80
        normal_result = normal_geometry.location(test_lon, test_lat)
        rotated_result = rotated_geometry.location(test_lon, test_lat)
        
        # The point's position should be different due to rotation
        assert (normal_result['delta_ra'] != rotated_result['delta_ra'] or 
                normal_result['delta_dec'] != rotated_result['delta_dec'])
        
        # But visibility and incidence angle should be unchanged
        assert normal_result['visible'] == rotated_result['visible']
        assert abs(normal_result['incoming'] - rotated_result['incoming']) < 0.01

    def test_location_label_formatting(self, mock_ephemeris):
        """Test the label formatting options in the location method."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Capture output with different max_title_len values
        outputs = []
        for max_len in [5, 10, 20]:
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                geometry.location(100, 20, label="TestLocation", max_title_len=max_len)
            outputs.append(captured_output.getvalue())
        
        # Check that all outputs contain the same numeric values
        for output in outputs:
            assert "LocalSolarTime:" in output
            assert "inc/emi ang:" in output
        
        # Verify that longer max_title_len allows for more space
        # (Comparing relative spacing rather than exact character positions)
        assert len(outputs[0].split("\n")[0]) < len(outputs[2].split("\n")[0])

    def test_longitude_wrapping(self, mock_ephemeris):
        """Test that longitude properly wraps around 360 degrees."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Test that equivalent longitudes yield the same results
        lon_pairs = [
            (10, 370),
            (-10, 350),
            (0, 360)
        ]
        
        for lon1, lon2 in lon_pairs:
            result1 = geometry.location(lon1, 30)
            result2 = geometry.location(lon2, 30)
            
            # All key parameters should be identical
            assert abs(result1['delta_ra'] - result2['delta_ra']) < 1e-10
            assert abs(result1['delta_dec'] - result2['delta_dec']) < 1e-10
            assert abs(result1['local_time'] - result2['local_time']) < 1e-10
            assert abs(result1['incoming'] - result2['incoming']) < 1e-10
            assert abs(result1['outgoing'] - result2['outgoing']) < 1e-10
            assert result1['visible'] == result2['visible']

    def test_visibility_edge_cases(self, mock_ephemeris):
        """Test visibility detection at the edge of the visible hemisphere."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Points around 90° from the sub-observer point should be at the visibility boundary
        sub_obs_lon = mock_ephemeris['ObsSub_LON']
        
        # Test points at same latitude, varying longitude
        results = []
        for lon_offset in range(0, 180, 15):
            test_lon = (sub_obs_lon + lon_offset) % 360
            result = geometry.location(test_lon, mock_ephemeris['ObsSub_LAT'])
            results.append((lon_offset, result['visible'], result['outgoing']))
        
        # Points with longitude offset < 90° should be visible
        for offset, visible, emission in results:
            if offset < 90:
                assert visible == True
                assert emission < 90
            elif offset > 90:
                assert visible == False or emission > 90

    def test_day_night_boundary(self, mock_ephemeris):
        """Test points at the day/night terminator (90° from sub-solar point)."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Points 90° from the sub-solar point should have ~90° incidence angles
        sub_solar_lon = mock_ephemeris['SunSub_LON']
        sub_solar_lat = mock_ephemeris['SunSub_LAT']
        
        # Test at equator with varying longitude
        for lon_offset in [90, -90]:
            test_lon = (sub_solar_lon + lon_offset) % 360
            result = geometry.location(test_lon, 0)  # At equator
            # Incidence angle should be close to 90°
            assert 80 <= result['incoming'] <= 100
        
        # Test at sub-solar longitude with varying latitude
        terminator_lat = sub_solar_lat + 90
        if -90 <= terminator_lat <= 90:  # Ensure latitude is valid
            result = geometry.location(sub_solar_lon, terminator_lat)
            assert 80 <= result['incoming'] <= 100

    def test_opposite_points(self, mock_ephemeris):
        """Test behavior at opposite points on the body."""
        geometry = BodyGeometry(mock_ephemeris)
        
        # Pick a visible point
        lon, lat = 110, 25
        result = geometry.location(lon, lat)
        assert result['visible'] == True
        
        # Test its opposite point
        opposite_lon = (lon + 180) % 360
        opposite_lat = -lat
        opposite_result = geometry.location(opposite_lon, opposite_lat)
        
        # Local time should differ by approximately 12 hours
        # Corregir el cálculo de diferencia horaria
        time_diff = abs(result['local_time'] - opposite_result['local_time'])
        if time_diff > 12:
            time_diff = 24 - time_diff
        assert abs(time_diff - 12) < 0.1
        
        # If the point is visible, its opposite point should have different visibility
        # or incidence/emission characteristics
        if opposite_result['visible']:
            # At least one of these should be true - different emission or incidence
            assert (abs(result['outgoing'] - opposite_result['outgoing']) > 10 or
                    abs(result['incoming'] - opposite_result['incoming']) > 10)
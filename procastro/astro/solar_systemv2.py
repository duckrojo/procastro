import logging
import re
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple, Union, Any

import requests
import numpy as np
from numpy import ma
from bs4 import BeautifulSoup
from tqdm import tqdm

from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib import pyplot as plt, transforms as mtransforms, collections as mcol, path as mpath
import cartopy.crs as ccrs
from PIL import Image

from astropy import time as apt, units as u, coordinates as apc, io as io
from astropy.table import Table, QTable, MaskedColumn

from procastro.astro.projection import new_x_axis_at, unit_vector, current_x_axis_to
from procastro.cache.cachev2 import jpl_cachev2, usgs_map_cachev2
from procastro.misc.misc_graph import figaxes

# Configuración del logger
logger = logging.getLogger("astro.solar_system")
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Tipos personalizados
TwoValues = Tuple[float, float]


class BodyGeometry:
    """
    Mantiene la geometría de un cuerpo celeste en un momento específico 
    con información retornada por JPL Horizons.

    Attributes
    ----------
    sub_obs : Tuple[float, float]
        Longitud y latitud en el cuerpo del punto sub-observador.
    sub_obs_np : float
        Ángulo del Polo Norte visto desde el punto sub-observador (Este de Norte).
    sub_sun : Tuple[float, float]
        Longitud y latitud en el cuerpo del punto sub-solar.
    ang_diam : float
        Diámetro angular del cuerpo en segundos de arco.
    """
    
    def __init__(self, ephemeris):
        """
        Inicializa la geometría del cuerpo a partir de datos de efemérides.
        
        Parameters
        ----------
        ephemeris : astropy.table.Row
            Fila de tabla de efemérides obtenida de JPL Horizons.
        """
        self._ephemeris = ephemeris
        self.sub_obs = ephemeris['ObsSub_LON'], ephemeris['ObsSub_LAT']
        self.sub_obs_np = ephemeris['NP_ang']
        self.sub_sun = ephemeris['SunSub_LON'], ephemeris['SunSub_LAT'],
        self.ang_diam = ephemeris['Ang_diam']

        self._rotate_to_subobs = new_x_axis_at(*self.sub_obs, z_pole_angle=-self.sub_obs_np)
        self._rotate_to_subsol = new_x_axis_at(*self.sub_sun)

    def print_info(self):
        """Imprime información detallada sobre la geometría del cuerpo."""
        print(f"Sub Observer longitude/latitude: {self._ephemeris['ObsSub_LON']}/{self._ephemeris['ObsSub_LAT']}")
        print(f"Sub Solar longitude/latitude: {self._ephemeris['SunSub_LON']}/{self._ephemeris['SunSub_LAT']}")
        print(f"Sub Solar angle/distances w/r to sub-observer: "
              f"{self._ephemeris['SN_ang']}/{self._ephemeris['SN_dist']}")
        print(f"North Pole angle/distances w/r to sub-observer: "
              f"{self._ephemeris['NP_ang']}/{self._ephemeris['NP_dist']}")

    def get_location_data(self, lon: float, lat: float, 
                         label: Optional[str] = None,
                         max_title_len: int = 0) -> Dict[str, Any]:
        """
        Calcula datos para una ubicación específica en la superficie del cuerpo.
        
        Parameters
        ----------
        lon : float
            Longitud en grados.
        lat : float
            Latitud en grados.
        label : str, optional
            Etiqueta para la ubicación (usado para impresión).
        max_title_len : int, optional
            Longitud máxima para formateo de títulos.
            
        Returns
        -------
        Dict[str, Any]
            Diccionario con datos de la ubicación (posición, tiempo local, ángulos, etc.).
        """
        position = unit_vector(lon, lat, degrees=True)

        unit_with_obs_x = self._rotate_to_subobs.apply(position)
        unit_with_sun_x = self._rotate_to_subsol.apply(position)

        delta_ra, delta_dec = self.ang_diam * unit_with_obs_x[1:] / 2
        local_time_location = (np.arctan2(*unit_with_sun_x[:2][::-1]) + np.pi) * 12 / np.pi
        incoming_angle = np.arccos(unit_with_sun_x[0]) * 180 / np.pi
        emission_angle = np.arccos(unit_with_obs_x[0]) * 180 / np.pi

        if label is not None:
            format_str = [f"{{name:{max_title_len + 1}s}} {{delta_ra:+10.2f}} {{delta_dec:+10.2f}}",
                         f"   (LocalSolarTime: {{local_time_location:+06.2f}}h, ",
                         f"inc/emi ang: {{incoming_angle:.0f}}/{{emission_angle:.0f}}deg)"]
            print("".join(format_str).format(name=str(label),
                                           local_time_location=local_time_location,
                                           delta_ra=delta_ra,
                                           delta_dec=delta_dec,
                                           emission_angle=emission_angle,
                                           incoming_angle=incoming_angle))
        
        return {
            'delta_ra': delta_ra,
            'delta_dec': delta_dec,
            'unit_ra': unit_with_obs_x[1],
            'unit_dec': unit_with_obs_x[2],
            'local_time': local_time_location,
            'incoming': incoming_angle,
            'outgoing': emission_angle,
            'visible': unit_with_obs_x[0] > 0,
        }


class JPLHorizonsQuery:
    """
    Clase para gestionar consultas a JPL Horizons y procesar sus resultados.
    """
    
    @staticmethod
    def _cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Producto vectorial entre dos vectores."""
        return np.cross(a, b)
    
    @staticmethod
    @jpl_cachev2
    def _request_horizons_online(specifications: str) -> List[str]:
        """
        Realiza una solicitud online a JPL Horizons.
        
        Parameters
        ----------
        specifications : str
            Especificación de la consulta para JPL Horizons.
            
        Returns
        -------
        List[str]
            Líneas de respuesta del servidor.
        
        Raises
        ------
        ValueError
            Si la respuesta del servidor no es válida.
        """
        # Implementación existente
        default_spec = {'MAKE_EPHEM': 'YES',
                       'EPHEM_TYPE': 'OBSERVER',
                       'CENTER': "'500@399'",
                       'QUANTITIES': "'1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,"
                                    "27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48'",
                       'REF_SYSTEM': "'ICRF'",
                       'CAL_FORMAT': "'JD'",
                       'CAL_TYPE': "'M'",
                       'TIME_DIGITS': "'MINUTES'",
                       'ANG_FORMAT': "'HMS'",
                       'APPARENT': "'AIRLESS'",
                       'RANGE_UNITS': "'AU'",
                       'SUPPRESS_RANGE_RATE': "'NO'",
                       'SKIP_DAYLT': "'NO'",
                       'SOLAR_ELONG': "'0,180'",
                       'EXTRA_PREC': "'NO'",
                       'R_T_S_ONLY': "'NO'",
                       'CSV_FORMAT': "'NO'",
                       'OBJ_DATA': "'YES'",
                       }
        # Resto del código existente...
        # ...
        custom_spec = {}
        prev = ""
        for spec in specifications.split("\n"):
            if spec[:6] == r"!$$SOF":
                continue
            kv = spec.strip().split("=")
            if len(kv) == 2:
                custom_spec[kv[0]] = kv[1]
                prev = kv[0]
            else:
                custom_spec[prev] += " " + kv[0]

        url_api = "https://ssd.jpl.nasa.gov/api/horizons.api?"
        full_specs = [f"{k}={v}" for k, v in (default_spec | custom_spec).items()
                      if k != 'TLIST']

        url = url_api + "&".join(full_specs)
        if 'TLIST' in custom_spec:
            url += f'&TLIST={custom_spec["TLIST"]}'
        
        try:
            if len(url) > 1000:
                if 'TLIST' in custom_spec:
                    epochs = custom_spec['TLIST'].split(' ')
                    epochs[0] = 'TLIST=' + epochs[0]
                    full_specs.extend(epochs)

                url_api_file = "https://ssd.jpl.nasa.gov/api/horizons_file.api?"
                with NamedTemporaryFile(mode="w", delete_on_close=False) as fp:
                    fp.write("!$$SOF\n")
                    fp.write("\n".join(full_specs))
                    fp.close()
                    response = requests.post(url_api_file,
                                          data={'format': 'text'},
                                          files={'input': open(fp.name)})
                    if response.status_code != 200:
                        raise ValueError(f"Error en solicitud a JPL Horizons: código {response.status_code}")
                    return response.text.splitlines()
            else:
                response = requests.get(url, allow_redirects=True)
                if response.status_code != 200:
                    raise ValueError(f"Error en solicitud a JPL Horizons: código {response.status_code}")
                return eval(response.content)['result'].splitlines()
        except Exception as e:
            logger.error(f"Error conectando con JPL Horizons: {str(e)}")
            raise
    
    @classmethod
    def get_ephemeris(cls, specification: Union[str, Dict[str, str]]) -> List[str]:
        """
        Obtiene datos de efemérides desde JPL Horizons.
        
        Parameters
        ----------
        specification : Union[str, Dict[str, str]]
            Especificación para la consulta. Puede ser un string o un diccionario.
            
        Returns
        -------
        List[str]
            Líneas de efemérides obtenidas.
            
        Raises
        ------
        FileNotFoundError
            Si el archivo especificado no existe.
        ValueError
            Si la especificación es inválida.
        """
        if isinstance(specification, dict):
            specification = f"""!$$SOF\n{"\n".join([f'{k}={v}' for k, v in specification.items()])}"""

        specification = specification.strip()
        if specification.count("\n") == 0:  # filename is given
            filename = Path(specification)
            if not filename.exists():
                raise FileNotFoundError(f"File '{filename}' does not exist")
            with open(filename, 'r') as fp:
                line = fp.readline()
                if line[:6] == r"!$$SOF":
                    ephemeris = cls._request_horizons_online(fp.read())
                else:
                    ephemeris = open(specification, 'r').readlines()
        else:
            if specification[:6] != r"!$$SOF":
                raise ValueError(f"Multiline Horizons specification invalid: {specification}")
            ephemeris = cls._request_horizons_online(specification)

        return ephemeris
    
    @classmethod
    def parse_ephemeris(cls, ephemeris: List[str]) -> Table:
        """
        Parsea líneas de efemérides en una tabla de astropy.
        
        Parameters
        ----------
        ephemeris : List[str]
            Líneas de efemérides de JPL Horizons.
            
        Returns
        -------
        astropy.table.Table
            Tabla con datos de efemérides procesados.
            
        Raises
        ------
        ValueError
            Si los datos de efemérides son inválidos.
        """
        # Implementación del parseo existente
        # ...
        # (Este método sería muy largo, mantendría la implementación existente
        # pero estructurada dentro de la clase)
        pass
    
    @classmethod
    def read_jpl(cls, specification: Union[str, Dict[str, str]]) -> Table:
        """
        Obtiene y parsea datos de efemérides de JPL Horizons.
        
        Parameters
        ----------
        specification : Union[str, Dict[str, str]]
            Especificación para la consulta.
            
        Returns
        -------
        astropy.table.Table
            Tabla con datos de efemérides procesados.
            
        Raises
        ------
        ValueError
            Si no se pueden obtener o procesar los datos de efemérides.
        """
        try:
            ephemeris = cls.get_ephemeris(specification)
            if ephemeris is None:
                raise ValueError("No se pudieron obtener datos de efemérides")
                
            table = cls.parse_ephemeris(ephemeris)
            if table is None:
                raise ValueError("No se pudieron procesar los datos de efemérides")
                
            return table
        except Exception as e:
            logger.error(f"Error en read_jpl: {str(e)}")
            raise
    
    @staticmethod
    def body_from_str(body: Union[str, int]) -> Dict[str, int]:
        """
        Convierte el nombre de un cuerpo en su ID numérico para JPL Horizons.
        
        Parameters
        ----------
        body : Union[str, int]
            Nombre del cuerpo o ID numérico.
            
        Returns
        -------
        Dict[str, int]
            Diccionario con el ID del cuerpo para JPL Horizons.
            
        Raises
        ------
        ValueError
            Si el cuerpo no es reconocido.
        """
        bodies = {
            'mercury': 199,
            'venus': 299,
            'moon': 301,
            'luna': 301,
            'mars': 499,
            'jupiter': 599,
            'saturn': 699,
            'uranus': 799,
            'neptune': 899,
        }
        
        if isinstance(body, int):
            return {'COMMAND': body}
        elif isinstance(body, str):
            try:
                body_id = bodies[body.lower()]
                return {'COMMAND': body_id}
            except KeyError:
                raise ValueError(f"Cuerpo desconocido: {body}")
        else:
            raise ValueError(f"Tipo de valor inválido para body: {type(body)}")
    
    @staticmethod
    def times_from_time(times: Union[str, apt.Time]) -> Dict[str, str]:
        """
        Genera especificación de tiempos para JPL Horizons.
        
        Parameters
        ----------
        times : Union[str, apt.Time]
            Tiempos para la consulta.
            
        Returns
        -------
        Dict[str, str]
            Diccionario con la especificación de tiempos.
            
        Raises
        ------
        ValueError
            Si hay demasiados tiempos discretos.
        """
        if isinstance(times, str):
            times = apt.Time(times, format='isot', scale='utc')
        if times.isscalar:
            times = apt.Time([times])
        if len(times) > 10000:
            raise ValueError("Horizon's interface only accepts a maximum of 10,000 discrete times")

        times_str = " ".join([f"'{s}'" for s in times.jd])
        return {
            'TLIST_TYPE': 'JD',
            'TIME_TYPE': 'UT',
            'TLIST': times_str
        }
    
    @staticmethod
    def observer_from_location(site: apc.EarthLocation) -> Dict[str, str]:
        """
        Genera especificación de observador para JPL Horizons.
        
        Parameters
        ----------
        site : apc.EarthLocation
            Ubicación del observador.
            
        Returns
        -------
        Dict[str, str]
            Diccionario con la especificación del observador.
        """
        return {
            'CENTER': "'coord@399'",
            'COORD_TYPE': "'GEODETIC'",
            'SITE_COORD': (f"'{site.lon.degree:.2f}, {site.lat.degree:.2f}," +
                           f" {site.height.to(u.km).value:.2f}'"),
        }


class USGSMapRetriever:
    """
    Clase para obtener y procesar mapas de USGS.
    """
    
    @staticmethod
    @usgs_map_cachev2
    def get_map_image(body: str, detail: Optional[str] = None, warn_multiple: bool = True) -> Optional[Image.Image]:
        """
        Obtiene una imagen de mapa de USGS para un cuerpo celeste.
        
        Parameters
        ----------
        body : str
            Nombre del cuerpo celeste.
        detail : str, optional
            Palabras clave para filtrar entre múltiples mapas disponibles.
        warn_multiple : bool, default=True
            Si se debe advertir cuando hay múltiples opciones disponibles.
            
        Returns
        -------
        PIL.Image.Image or None
            Imagen del mapa, o None si no se pudo obtener.
            
        Raises
        ------
        ValueError
            Si no se encuentra ningún mapa para el cuerpo especificado.
        """
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        def _parse_date(string):
            if string is None:
                return ""
            if len(string) == 8:
                return f"{string[:4]}-{month[int(string[4:6])-1]}{string[6:8]}"
            raise ValueError(f"Needs to implement parsing for date: {string}")

        directory = (Path(__file__).parents[0] / 'images')
        files = list(directory.glob("*.xml"))

        keywords = None
        if detail is not None:
            keywords = detail.split()

        # filter alternatives
        body_files = []
        for file in files:
            try:
                with open(file, 'r', encoding='utf8') as f:
                    data = BeautifulSoup(f.read(), 'xml')
                    body_in_xml = data.find("target").string
                    if body.lower() == body_in_xml.lower():
                        title = data.idinfo("title")[0].string
                        if keywords is not None and not [k for k in keywords if k in title]:
                            continue

                        info = [title,
                               file,
                               data.find("browsen").string,
                               data.idinfo("begdate")[0].string,
                               data.idinfo("enddate")[0].string,
                               ]
                        if 'default' in str(file):
                            body_files.insert(0, info)
                        else:
                            body_files.append(info)
            except Exception as e:
                logger.warning(f"Error processing XML file {file}: {str(e)}")
                continue

        detail_str = f" with detail keywords '{detail}'" if detail else ""
        if len(body_files) == 0:
            raise ValueError(f"No map of '{body}' found{detail_str}")
        
        # select from alternatives
        elif len(body_files) > 1 and warn_multiple:
            suggest = "" if detail else " (use space-separated keywords in 'detail' to filter).\n"
            print(f"Several map alternatives for {body} were available{detail_str}\n{suggest}" 
                  "Selected first of:")
            for bf in body_files:
                print(f"* {bf[0]} [{_parse_date(bf[3])}..{_parse_date(bf[4])}]")
            print("")

        # Seleccionar el primer mapa (posiblemente el único o el predeterminado)
        selected_map = body_files[0]
        
        # Descargar la imagen
        logger.info(f"HTTP GET REQUEST TO: {selected_map[2]}")
        try:
            response = requests.get(selected_map[2], timeout=30)
            if response.status_code != 200:
                logger.error(f"Error fetching USGS map: HTTP {response.status_code}")
                return None
                
            return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            logger.error(f"Network error fetching USGS map: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing USGS map: {str(e)}")
            return None

class ProjectionUtils:
    """
    Utilitarios para la proyección y visualización de cuerpos celestes.
    """
    
    @staticmethod
    def get_orthographic(platecarree_image: Image.Image,
                         sub_obs_lon: float,
                         sub_obs_lat: float,
                         marks: Optional[List[Tuple[float, float]]] = None,
                         show_poles: str = "") -> Image.Image:
        """
        Genera una proyección ortográfica de una imagen de mapa.
        
        Parameters
        ----------
        platecarree_image : PIL.Image.Image
            Imagen del mapa en proyección platecarree.
        sub_obs_lon : float
            Longitud del punto sub-observador.
        sub_obs_lat : float
            Latitud del punto sub-observador.
        marks : List[Tuple[float, float]], optional
            Lista de coordenadas (lon, lat) para marcar en el mapa.
        show_poles : str, default=""
            Color para marcar los polos. Cadena vacía para no mostrarlos.
            
        Returns
        -------
        PIL.Image.Image
            Imagen en proyección ortográfica.
        """
        projection = ccrs.Orthographic(sub_obs_lon, sub_obs_lat)

        # Guardar el backend actual
        backend = plt.get_backend()
        plt.switch_backend('agg')

        f = plt.figure(figsize=(3, 3))
        tmp_ax = f.add_axes((0, 0, 1, 1),
                           transform=f.transFigure,
                           projection=projection)

        tmp_ax.imshow(platecarree_image,
                     origin='upper',
                     transform=ccrs.PlateCarree(),
                     extent=(-180, 180, -90, 90),
                     )
        
        if show_poles:
            tmp_ax.plot(0, 90,
                       transform=ccrs.PlateCarree(),
                       marker='d', color=show_poles)
            tmp_ax.plot(0, -90,
                       transform=ccrs.PlateCarree(),
                       marker='d', color=show_poles)

        if marks is not None:
            for mark in marks:
                tmp_ax.plot(*mark,
                          transform=ccrs.PlateCarree(),
                          marker='x', color=show_poles)

        tmp_ax.axis('off')
        tmp_ax.set_global()  # Mostrar todo el globo
        f.canvas.draw()

        # Convertir a imagen
        image_flat = np.frombuffer(f.canvas.tostring_argb(), dtype='uint8')
        orthographic_image = image_flat.reshape(*reversed(f.canvas.get_width_height()), 4)[:, :, 1:]
        orthographic_image = Image.fromarray(orthographic_image, 'RGB')
        
        plt.close(f)
        plt.switch_backend(backend)

        return orthographic_image
    
    @staticmethod
    def add_local_time(ax: Axes,
                      sub_obs: TwoValues,
                      sub_sun: TwoValues,
                      np_ang: float,
                      color: str,
                      transform_from_norm,
                      precision: int = 50) -> Tuple:
        """
        Añade marcas de tiempo local al mapa del cuerpo celeste.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Ejes donde dibujar.
        sub_obs : Tuple[float, float]
            Coordenadas del punto sub-observador.
        sub_sun : Tuple[float, float]
            Coordenadas del punto sub-solar.
        np_ang : float
            Ángulo del polo norte.
        color : str
            Color para las líneas y etiquetas de tiempo local.
        transform_from_norm
            Transformación de coordenadas.
        precision : int, default=50
            Precisión para el cálculo de líneas.
            
        Returns
        -------
        Tuple
            Tupla con los artistas creados.
        """
        lunar_to_observer = new_x_axis_at(*sub_obs, z_pole_angle=-np_ang)
        local_time_to_body = current_x_axis_to(*sub_sun)

        artists = []
        for ltime in range(24):
            longitude_as_local_time = new_x_axis_at((12-ltime)*15, 0)

            local_time_rotation = lunar_to_observer * local_time_to_body * longitude_as_local_time
            local_time = local_time_rotation.apply(unit_vector(0, np.linspace(-90, 90, precision),
                                                             degrees=True))
            
            # No plotear arcos que están en el lado oculto del objeto
            visible = np.array([(y, z) for x, y, z in local_time if x > 0])
            n_visible = len(visible)
            
            if n_visible > precision//2:  # Si más de la mitad del arco es visible
                lines = ax.plot(visible[:, 0], visible[:, 1],
                              color=color,
                              alpha=0.7,
                              ls=':',
                              transform=transform_from_norm,
                              zorder=6,
                              linewidth=0.5)

                text = ax.annotate(f"{ltime}$^h$", (visible[n_visible // 2][0],
                                                   visible[n_visible // 2][1]),
                                 color=color,
                                 xycoords=transform_from_norm,
                                 alpha=0.7, ha='center', va='center')

                artists.extend([lines, text])

        return tuple(artists)
    
    @staticmethod
    def add_phase_shadow(ax: Axes,
                        sub_obs: TwoValues,
                        sub_sun: TwoValues,
                        np_ang: float,
                        color: str,
                        transform_from_normal,
                        precision: int = 50,
                        marker_color: str = 'yellow') -> Tuple:
        """
        Añade sombra de fase al mapa del cuerpo celeste.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Ejes donde dibujar.
        sub_obs : Tuple[float, float]
            Coordenadas del punto sub-observador.
        sub_sun : Tuple[float, float]
            Coordenadas del punto sub-solar.
        np_ang : float
            Ángulo del polo norte.
        color : str
            Color para la sombra de fase.
        transform_from_normal
            Transformación de coordenadas.
        precision : int, default=50
            Precisión para el cálculo de líneas.
        marker_color : str, default='yellow'
            Color para el marcador del punto sub-solar.
            
        Returns
        -------
        Tuple
            Tupla con los artistas creados.
        """
        # Implementar lógica de añadir sombra de fase
        rotate_body_to_subobs = new_x_axis_at(*sub_obs, z_pole_angle=-np_ang)
        rotate_body_to_subsol = new_x_axis_at(*sub_sun)

        upper_vector_terminator = np.cross(unit_vector(*sub_obs), unit_vector(*sub_sun))
        upper_shadow_from_sun = new_x_axis_at(*sub_sun).apply(upper_vector_terminator)
        upper_angle_from_sun = (np.arctan2(*upper_shadow_from_sun[1:][::-1]) - np.pi / 2) * 180 / np.pi

        starting_terminator = new_x_axis_at(0, 90).apply(unit_vector(np.linspace(0, 360, precision), 0))
        terminator_at_body = current_x_axis_to(*sub_sun, z_pole_angle=-upper_angle_from_sun).apply(starting_terminator)
        terminator_sub_obs = np.array([(y, z) for x, y, z in rotate_body_to_subobs.apply(terminator_at_body) if x > 0])

        delta = ((terminator_sub_obs[1:,0] - terminator_sub_obs[:-1,0])**2 +
                (terminator_sub_obs[1:,1] - terminator_sub_obs[:-1,1])**2)
        max_delta = np.argmax(delta)
        if delta[max_delta] > 4/precision:
            terminator_sub_obs = np.roll(terminator_sub_obs, -(max_delta+1), axis=0)

        angle_first, angle_last = np.arctan2(*np.array(terminator_sub_obs)[[0, -1]].transpose()[::-1])

        if angle_first > angle_last:
            angle_last += 2*np.pi

        angle_perimeter = np.linspace(angle_last, angle_first, precision)
        perimeter = np.array([np.array([np.cos(ang), np.sin(ang)])
                             for ang in angle_perimeter]
                            + list(terminator_sub_obs))

        clip_path = mpath.Path(vertices=perimeter, closed=True)

        col = mcol.PathCollection([clip_path],
                                 facecolors=color, alpha=0.7,
                                 edgecolors='none',
                                 zorder=7,
                                 transform=transform_from_normal)
        ax.add_collection(col)

        projected_sub_sun = rotate_body_to_subobs.apply(unit_vector(*sub_sun))
        sub_sun_marker = ax.plot(*projected_sub_sun[1:],
                                marker='d', color=marker_color,
                                alpha=1 - 0.5 * (projected_sub_sun[0] < 0),
                                transform=transform_from_normal)
        ax.annotate(f"{np.abs(sub_sun[1]):.1f}$^\\circ${'N' if sub_sun[1] > 0 else 'S'}",
                   projected_sub_sun[1:],
                   xycoords=transform_from_normal,
                   color=marker_color)

        return col, sub_sun_marker


class BodyMapGenerator:
    """
    Clase para generar mapas de cuerpos celestes.
    """
    
    def __init__(self):
        """Inicializa el generador de mapas."""
        self.utils = ProjectionUtils()
    
    def generate_map(self,
                    body: str,
                    observer: Union[str, Table.Row],
                    time: Optional[apt.Time] = None,
                    locations: Optional[Union[List[TwoValues], Dict[str, TwoValues]]] = None,
                    title: Optional[str] = None,
                    detail: Optional[str] = None,
                    reread_usgs: bool = False,
                    radius_to_plot: Optional[float] = None,
                    ax: Optional[Axes] = None,
                    return_axes: bool = True,
                    verbose: bool = True,
                    color_location: str = "red",
                    color_phase: str = 'black',
                    color_background: str = 'black',
                    color_title: str = 'white',
                    color_local_time: str = 'black',
                    color_poles: str = "blue",
                    show_angles: bool = False,
                    ) -> Union[Axes, List]:
        """
        Genera un mapa de un cuerpo celeste.
        
        Parameters
        ----------
        body : str
            Nombre del cuerpo celeste.
        observer : str or astropy.table.Row
            Ubicación del observador o fila de efemérides.
        time : apt.Time, optional
            Tiempo de observación. Requerido si observer no es una fila de efemérides.
        locations : List[TwoValues] or Dict[str, TwoValues], optional
            Ubicaciones a marcar en el mapa.
        title : str, optional
            Título para el mapa.
        detail : str, optional
            Palabras clave para seleccionar un mapa específico.
        reread_usgs : bool, default=False
            Si se debe releer el mapa de USGS incluso si existe en caché.
        radius_to_plot : float, optional
            Radio a mostrar en segundos de arco.
        ax : matplotlib.axes.Axes, optional
            Ejes donde dibujar el mapa.
        return_axes : bool, default=True
            Si se deben devolver los ejes o los artistas.
        verbose : bool, default=True
            Si se debe imprimir información detallada.
        color_location : str, default="red"
            Color para las ubicaciones marcadas.
        color_phase : str, default='black'
            Color para la sombra de fase.
        color_background : str, default='black'
            Color de fondo.
        color_title : str, default='white'
            Color del título.
        color_local_time : str, default='black'
            Color para las marcas de tiempo local.
        color_poles : str, default="blue"
            Color para los polos.
        show_angles : bool, default=False
            Si se deben mostrar ángulos de incidencia y emisión.
            
        Returns
        -------
        matplotlib.axes.Axes or List
            Ejes o lista de artistas, según el valor de return_axes.
            
        Raises
        ------
        TypeError
            Si time es None y observer no es una fila de efemérides.
        ValueError
            Si no se puede obtener la imagen del cuerpo.
        """
        # Verificar tipo de observer y obtener efemérides
        if time is None:
            if not isinstance(observer, Table.Row):
                raise TypeError("Time can only be omitted when observer is a JPL ephemeris in a Table.Row object.")
            ephemeris_line = observer
            time = apt.Time(ephemeris_line['jd'], format='jd')
        else:
            # Verificar si time es escalar, si no lo es, delegar a BodyMapAnimator
            if not time.isscalar:
                logger.info("Multiple times detected, delegating to BodyMapAnimator")
                animator = BodyMapAnimator()
                return animator.generate_animation(
                    body, observer, time,
                    locations=locations,
                    title=title,
                    detail=detail,
                    reread_usgs=reread_usgs,
                    radius_to_plot=radius_to_plot,
                    ax=ax,
                    color_location=color_location,
                    color_phase=color_phase,
                    color_background=color_background,
                    color_title=color_title,
                    color_local_time=color_local_time,
                    color_poles=color_poles,
                    show_angles=show_angles
                )
            
            # Obtener efemérides para un tiempo específico
            site = apc.EarthLocation.of_site(observer)
            request = (JPLHorizonsQuery.observer_from_location(site) | 
                      JPLHorizonsQuery.times_from_time(time) | 
                      JPLHorizonsQuery.body_from_str(body))
            try:
                ephemeris_line = JPLHorizonsQuery.read_jpl(request)[0]
            except Exception as e:
                logger.error(f"Error obtaining ephemeris: {str(e)}")
                raise

        # Obtener geometría del cuerpo
        geometry = BodyGeometry(ephemeris_line)
        if verbose:
            geometry.print_info()

        # Obtener imagen del mapa
        try:
            image = USGSMapRetriever.get_map_image(body, detail=detail, warn_multiple=verbose)
            if image is None:
                raise ValueError(f"Could not get image for {body}")
        except Exception as e:
            logger.error(f"Error getting map image: {str(e)}")
            raise

        # Generar imagen ortográfica
        try:
            orthographic_image = self.utils.get_orthographic(
                image, *geometry.sub_obs, show_poles=color_poles
            )
        except Exception as e:
            logger.error(f"Error generating orthographic projection: {str(e)}")
            raise

        # Rotar imagen según el ángulo del polo norte
        try:
            rotated_image = orthographic_image.rotate(
                geometry.sub_obs_np,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(255, 255, 255)
            )
        except Exception as e:
            logger.error(f"Error rotating image: {str(e)}")
            raise

        # Aplicar transparencia para el fondo
        x_offset = 0.5
        y_offset = 0
        ny, nx = rotated_image.size
        yy, xx = np.mgrid[-ny/2 + y_offset: ny/2 + y_offset, -nx/2 + x_offset: nx/2 + x_offset]
        rr = np.sqrt(yy**2 + xx**2).flatten()

        color_background_rgb = (*[int(c*255) for c in to_rgba(color_background)[:3]], 0)
        rotated_image.putdata([item if r < nx/2 - 1 else color_background_rgb
                              for r, item in zip(rr, rotated_image.convert("RGBA").getdata())])

        # Crear figura y ejes si no se proporcionan
        f, ax = figaxes(ax)
        f.patch.set_facecolor(color_background)
        ax.set_facecolor(color_background)
        ax.imshow(rotated_image)
        ax.axis('off')

        # Configurar límites y mostrar imagen
        ang_rad = geometry.ang_diam/2
        ax.imshow(rotated_image, extent=[-ang_rad, ang_rad, -ang_rad, ang_rad])

        if radius_to_plot is None:
            radius_to_plot = ang_rad

        ax.set_xlim([-radius_to_plot, radius_to_plot])
        ax.set_ylim([-radius_to_plot, radius_to_plot])

        # Añadir título
        if color_title is not None and color_title:
            if title is None:
                radius = ang_rad
                rad_unit = '"'
                if radius > 120:
                    radius /= 60
                    rad_unit = "'"
                title = f'{body.capitalize()} on {time.isot[:16]} (R$_{body[0].upper()}$: {radius:.1f}{rad_unit})'
            ax.set_title(title, color=color_title)

        # Transformación para coordenadas normalizadas
        transform_norm_to_axes = mtransforms.Affine2D().scale(ang_rad) + ax.transData

        # Añadir ubicaciones
        if locations is not None and len(locations) > 0:
            if verbose:
                print(f"Location offset from {body.capitalize()}'s center (Delta_RA, Delta_Dec) in arcsec:")

            if isinstance(locations, list):
                locations = {str(i): v for i, v in enumerate(locations)}
            
            max_len = max([len(str(k)) for k in locations.keys()])
            
            for name, location in locations.items():
                try:
                    loc_data = geometry.get_location_data(
                        *location, 
                        label=name if verbose else None,
                        max_title_len=max_len
                    )

                    incoming_emission = f"{loc_data['incoming']:.0f}/{loc_data['outgoing']:.0f}"
                    ie_plot = f", i/e {incoming_emission}$^{{\\circ}}$" if show_angles else ""

                    ax.plot(
                        loc_data['unit_ra'], loc_data['unit_dec'],
                        transform=transform_norm_to_axes,
                        marker='d', color=color_location,
                        alpha=1 - 0.5 * (not loc_data['visible']),
                        zorder=10
                    )

                    ax.annotate(
                        f"{str(name)}: $\\Delta\\alpha$ {loc_data['delta_ra']:+.0f}\", "
                        f"$\\Delta\\delta$ {loc_data['delta_dec']:.0f}\", "
                        f"LT{loc_data['local_time']:04.1f}$^h${ie_plot}",
                        (loc_data['unit_ra'], loc_data['unit_dec']),
                        xycoords=transform_norm_to_axes,
                        color=color_location,
                        alpha=1 - 0.5 * (not loc_data['visible']),
                        zorder=10
                    )
                except Exception as e:
                    logger.warning(f"Error adding location {name}: {str(e)}")

            if verbose:
                print("")

        # Añadir ejes y polos
        if color_poles is not None and color_poles:
            ax.plot(
                [-1, 1], [0, 0],
                color='blue',
                transform=transform_norm_to_axes,
                zorder=9
            )
            ax.plot(
                [0, 0], [-1, 1],
                color='blue',
                transform=transform_norm_to_axes,
                zorder=9
            )

            for lat_pole in [-90, 90]:
                try:
                    pole = geometry.get_location_data(0, lat_pole)
                    ax.plot(
                        pole['unit_ra'], pole['unit_dec'],
                        transform=transform_norm_to_axes,
                        alpha=1 - 0.5 * (not pole['visible']),
                        marker='o', color='blue',
                        zorder=9
                    )
                except Exception as e:
                    logger.warning(f"Error adding pole at latitude {lat_pole}: {str(e)}")

        # Añadir marcas de tiempo local
        if color_local_time:
            try:
                self.utils.add_local_time(
                    ax,
                    geometry.sub_obs,
                    geometry.sub_sun,
                    geometry.sub_obs_np,
                    color_phase,
                    transform_norm_to_axes
                )
            except Exception as e:
                logger.warning(f"Error adding local time markers: {str(e)}")

        # Añadir sombra de fase
        if color_phase is not None and color_phase:
            try:
                self.utils.add_phase_shadow(
                    ax,
                    geometry.sub_obs,
                    geometry.sub_sun,
                    geometry.sub_obs_np,
                    color_phase,
                    transform_norm_to_axes
                )
            except Exception as e:
                logger.warning(f"Error adding phase shadow: {str(e)}")

        if return_axes:
            return ax
        else:
            return ax.collections + ax.lines + ax.texts + ax.images


class BodyMapAnimator:
    """
    Clase para generar animaciones de cuerpos celestes.
    """
    
    def __init__(self):
        """Inicializa el generador de animaciones."""
        self.map_generator = BodyMapGenerator()
    
    def create_frame(self,
                   body: str,
                   ephemeris: Table.Row,
                   time_label: str,
                   ax: Axes,
                   title_template: Optional[str] = None,
                   field: str = 'Ang_diam',
                   divider: float = 1.0,
                   **kwargs) -> List:
        """
        Crea un frame para la animación.
        
        Parameters
        ----------
        body : str
            Nombre del cuerpo celeste.
        ephemeris : astropy.table.Row
            Fila de efemérides.
        time_label : str
            Etiqueta de tiempo para el frame.
        ax : matplotlib.axes.Axes
            Ejes donde dibujar el frame.
        title_template : str, optional
            Plantilla para el título.
        field : str, default='Ang_diam'
            Campo de efemérides a mostrar en el título.
        divider : float, default=1.0
            Divisor para el valor del campo.
        **kwargs
            Argumentos adicionales para generate_map.
            
        Returns
        -------
        List
            Lista de artistas creados.
        """
        ax.clear()
        
        title = None
        if title_template:
            title = title_template.format(
                time=time_label,
                field=ephemeris[field]/divider
            )
        
        # Usar el generador de mapas para crear el frame
        return self.map_generator.generate_map(
            body,
            ephemeris,
            return_axes=False,
            verbose=False,
            title=title,
            ax=ax,
            **kwargs
        )
    
    def generate_animation(self,
                         body: str,
                         observer: str,
                         times: apt.Time,
                         filename: Optional[str] = None,
                         fps: int = 10,
                         dpi: Optional[int] = 75,
                         title: Optional[str] = None,
                         ax: Optional[Axes] = None,
                         color_background: str = "black",
                         radius_to_plot: Optional[float] = None,
                         **kwargs) -> None:
        """
        Genera una animación de un cuerpo celeste a lo largo del tiempo.
        
        Parameters
        ----------
        body : str
            Nombre del cuerpo celeste.
        observer : str
            Ubicación del observador.
        times : apt.Time
            Tiempos para la animación.
        filename : str, optional
            Nombre de archivo para guardar la animación.
        fps : int, default=10
            Frames por segundo.
        dpi : int, optional, default=75
            Resolución en puntos por pulgada.
        title : str, optional
            Plantilla para el título.
        ax : matplotlib.axes.Axes, optional
            Ejes donde dibujar la animación.
        color_background : str, default="black"
            Color de fondo.
        radius_to_plot : float, optional
            Radio a mostrar en segundos de arco.
        **kwargs
            Argumentos adicionales para generate_map.
            
        Raises
        ------
        ValueError
            Si hay algún error durante la generación de la animación.
        FileNotFoundError
            Si el codec para el formato de archivo no está disponible.
        """
        print("Creating animation...")
        
        # Guardar el backend actual
        backend = plt.get_backend()
        plt.switch_backend('agg')

        # Crear figura y ejes si no se proporcionan
        if ax is None:
            f, ax = figaxes()
        else:
            f = ax.figure

        f.set_facecolor(color_background)

        try:
            # Obtener efemérides para todos los tiempos
            site = apc.EarthLocation.of_site(observer)
            request = (JPLHorizonsQuery.observer_from_location(site) | 
                      JPLHorizonsQuery.times_from_time(times) | 
                      JPLHorizonsQuery.body_from_str(body))
            
            try:
                ephemeris_lines = JPLHorizonsQuery.read_jpl(request)

            except Exception as e:
                logger.error(f"Error obtaining ephemeris: {str(e)}")
                raise ValueError(f"Could not obtain ephemeris for {body} at {observer}")

            field = 'Ang_diam'

            if radius_to_plot is None:
                radius_to_plot = np.max(ephemeris_lines[field])/2

            # Determinar la unidad y divisor apropiados para el radio
            if title is None:
                mean_rad = np.mean(ephemeris_lines[field])
                if mean_rad > 120:
                    rad_unit = "'"
                    divider = 60
                elif mean_rad > 1:
                    rad_unit = '"'
                    divider = 1
                else:
                    rad_unit = 'mas'
                    divider = 0.001
                title = f'{body.capitalize()} on {{time}} from {observer} (R$_{body[0].upper()}$: {{field:.1f}}{rad_unit})'

            # Crear barra de progreso
            pbar = tqdm(total=len(times), desc="Animating")
            
            # Función para crear cada frame
            def animate(itime):
                ephemeris = ephemeris_lines[itime]
                y, m, d, h, mn, s = times[itime].ymdhms
                time_label = f"{y}.{m:02d}.{d:02d} {h:02d}:{mn+s/60:04.1f} UT"
                
                artists = self.create_frame(
                    body, ephemeris, time_label, ax,
                    title_template=title,
                    field=field,
                    divider=divider,
                    radius_to_plot=radius_to_plot,
                    color_background=color_background,
                    **kwargs
                )
                
                pbar.update(1)
                return artists

            # Crear animación
            ani = FuncAnimation(
                f, animate, interval=60, blit=False, 
                repeat=True, frames=len(times)
            )
            
            # Determinar nombre y tipo de archivo
            if filename is None:
                filename = f"{body}_{times[0].isot.replace(':', '')[:17]}.gif"
            elif '.' not in filename:
                filename += '.gif'
            
            # Seleccionar escritor según extensión
            extension = filename[filename.index('.'):].lower()
            if extension in ('.mpg', '.mpeg', '.mp4'):
                writer = FFMpegWriter(fps=fps)
            elif extension == '.gif':
                writer = PillowWriter(fps=fps)
            else:
                raise ValueError(f"Invalid filename extension: {extension}")

            # Guardar animación
            ani.save(
                filename,
                dpi=dpi,
                writer=writer
            )
            
            print(f"Saved animation to {filename}")
            
        except FileNotFoundError as e:
            logger.error(f"Codec not available: {str(e)}")
            raise FileNotFoundError(f"Codec for extension {filename[filename.index('.'):]} not available.")
        except Exception as e:
            logger.error(f"Error generating animation: {str(e)}")
            raise
        finally:
            # Restaurar backend y cerrar barra de progreso
            plt.switch_backend(backend)
            if 'pbar' in locals():
                pbar.close()


def body_mapv2(body: str,
            observer: Union[str, Table.Row],
            time: Optional[apt.Time] = None,
            locations: Optional[Union[List[TwoValues], Dict[str, TwoValues]]] = None,
            title: Optional[str] = None,
            detail: Optional[str] = None,
            reread_usgs: bool = False,
            radius_to_plot: Optional[float] = None,
            fps: int = 10,
            dpi: int = 75,
            filename: Optional[str] = None,
            ax: Optional[Axes] = None,
            return_axes: bool = True,
            verbose: bool = True,
            color_location: str = "red",
            color_phase: str = 'black',
            color_background: str = 'black',
            color_title: str = 'white',
            color_local_time: str = 'black',
            color_poles: str = "blue",
            show_angles: bool = False,
            ) -> Optional[Union[Axes, List]]:
    """
    Genera un mapa o animación de un cuerpo celeste.
    
    Esta función actúa como interfaz principal para la generación de mapas y
    animaciones de cuerpos celestes, delegando a las clases especializadas.
    
    Parameters
    ----------
    body : str
        Nombre del cuerpo celeste.
    observer : str or astropy.table.Row
        Ubicación del observador o fila de efemérides.
    time : apt.Time, optional
        Tiempo de observación. Si es None, observer debe ser una fila de efemérides.
        Si contiene múltiples tiempos, se genera una animación.
    locations : List[TwoValues] or Dict[str, TwoValues], optional
        Ubicaciones a marcar en el mapa.
    title : str, optional
        Título para el mapa o animación.
    detail : str, optional
        Palabras clave para seleccionar un mapa específico.
    reread_usgs : bool, default=False
        Si se debe releer el mapa de USGS incluso si existe en caché.
    radius_to_plot : float, optional
        Radio a mostrar en segundos de arco.
    fps : int, default=10
        Frames por segundo para animaciones.
    dpi : int, default=75
        Resolución en puntos por pulgada para animaciones.
    filename : str, optional
        Nombre de archivo para guardar la animación.
    ax : matplotlib.axes.Axes, optional
        Ejes donde dibujar el mapa o animación.
    return_axes : bool, default=True
        Si se deben devolver los ejes o los artistas.
    verbose : bool, default=True
        Si se debe imprimir información detallada.
    color_location : str, default="red"
        Color para las ubicaciones marcadas.
    color_phase : str, default='black'
        Color para la sombra de fase.
    color_background : str, default='black'
        Color de fondo.
    color_title : str, default='white'
        Color del título.
    color_local_time : str, default='black'
        Color para las marcas de tiempo local.
    color_poles : str, default="blue"
        Color para los polos.
    show_angles : bool, default=False
        Si se deben mostrar ángulos de incidencia y emisión.
        
    Returns
    -------
    matplotlib.axes.Axes or List or None
        Ejes, lista de artistas, o None en caso de animación.
        
    Notes
    -----
    Esta función es el punto de entrada principal para la generación de mapas
    de cuerpos celestes y animaciones de su movimiento a lo largo del tiempo.
    """
    if time is not None and not time.isscalar:
        # Crear una animación con múltiples tiempos
        animator = BodyMapAnimator()
        animator.generate_animation(
            body, observer, time,
            locations=locations,
            title=title,
            detail=detail, 
            reread_usgs=reread_usgs,
            radius_to_plot=radius_to_plot,
            fps=fps, 
            dpi=dpi, 
            filename=filename,
            ax=ax,
            color_location=color_location,
            color_phase=color_phase,
            color_background=color_background,
            color_title=color_title,
            color_local_time=color_local_time,
            color_poles=color_poles,
            show_angles=show_angles
        )
        return None
    else:
        # Crear un mapa para un solo tiempo
        generator = BodyMapGenerator()
        return generator.generate_map(
            body, observer, time,
            locations=locations,
            title=title,
            detail=detail,
            reread_usgs=reread_usgs,
            radius_to_plot=radius_to_plot,
            ax=ax,
            return_axes=return_axes,
            verbose=verbose,
            color_location=color_location,
            color_phase=color_phase,
            color_background=color_background,
            color_title=color_title,
            color_local_time=color_local_time,
            color_poles=color_poles,
            show_angles=show_angles
        )


def body_path(body: str,
             observer: str,
             times: apt.Time,
             use_jpl: bool = False) -> QTable:
    """
    Obtiene la trayectoria de un cuerpo celeste a lo largo del tiempo.
    
    Parameters
    ----------
    body : str
        Nombre del cuerpo celeste.
    observer : str
        Ubicación del observador.
    times : apt.Time
        Tiempos para los que calcular la trayectoria.
    use_jpl : bool, default=False
        Si se debe usar JPL para los cálculos.
        
    Returns
    -------
    astropy.table.QTable
        Tabla con la trayectoria del cuerpo.
    """
    if use_jpl:
        site = apc.EarthLocation.of_site(observer)
        request = (JPLHorizonsQuery.observer_from_location(site) | 
                  JPLHorizonsQuery.times_from_time(times) | 
                  JPLHorizonsQuery.body_from_str(body))
        
        ret = JPLHorizonsQuery.read_jpl(request)
        ret['skycoord'] = apc.SkyCoord(ret['ra_ICRF'], ret['dec_ICRF'], unit=(u.hourangle, u.degree))
        return ret
    else:
        site = apc.EarthLocation.of_site(observer)
        body_object = apc.get_body(body, times, location=site)

        return QTable([times, times.jd, body_object,
                      body_object.ra.degree, body_object.dec.degree],
                     names=['time', 'jd', 'skycoord', 'ra', 'dec'])
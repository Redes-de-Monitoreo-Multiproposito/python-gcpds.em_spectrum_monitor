import xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2
from typing import List


########################################################################
class Stations:
    """"""

    # ----------------------------------------------------------------------
    def __init__(self):
        """"""

        # Load and parse the XML file
        xml_file_path = 'xml-propertyvalue-754348.xml'
        tree = ET.parse(xml_file_path)
        self.root = tree.getroot()

    # ----------------------------------------------------------------------
    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two points
        on the Earth using the Haversine formula.

        Parameters
        ----------
        lat1 : float
            Latitude of the first point in degrees.
        lon1 : float
            Longitude of the first point in degrees.
        lat2 : float
            Latitude of the second point in degrees.
        lon2 : float
            Longitude of the second point in degrees.

        Returns
        -------
        float
            The distance between the two points in kilometers.

        Notes
        -----
        The Haversine formula is used to calculate the distance between
        two points on the surface of a sphere given their longitudes
        and latitudes. The formula accounts for the spherical shape of
        the Earth to provide an accurate measurement of distance.
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula to calculate the distance
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        r = 6371  # Radius of Earth in kilometers
        return r * c


    # ----------------------------------------------------------------------
    def find_nearest_stations(self, lat: float, lon: float, n: int = 5) -> List[dict]:
        """
        Find the nearest radio stations to the given latitude and longitude.

        This function calculates the distance from the given `lat` and `lon`
        coordinates to each radio station in the XML data using the Haversine
        formula. It then returns a list of dictionaries containing details
        about the `n` nearest stations.

        Parameters
        ----------
        lat : float
            Latitude of the location to search from.
        lon : float
            Longitude of the location to search from.
        n : int, optional
            Number of nearest stations to return. Default is 5.

        Returns
        -------
        List[dict]
            A list of dictionaries containing details of the nearest stations.
            Each dictionary contains the following keys:
                - 'nombre'
                - 'organizacion'
                - 'location'
                - 'concesionario'
                - 'codigo'
                - 'frecuencia'
                - 'distintivo'
                - 'tecnologia'
                - 'clase_programa'
                - 'latlong'
                - 'departamento'
                - 'distance'
        """
        stations = []

        for emisora in self.root.findall('radioemisora'):
            try:
                latlong = emisora.find('latlong').text
                station_lat, station_lon = map(float, latlong.split(','))
                distance = self.haversine(lat, lon, station_lat, station_lon)
                stations.append((distance, emisora))
            except (AttributeError, ValueError):
                continue

        # Sort stations by distance
        stations.sort(key=lambda x: x[0])

        # Extract the closest n stations
        nearest_stations = stations[:n]

        # Format the result
        result = []
        for distance, emisora in nearest_stations:
            station_info = {
                'nombre': emisora.find('nombre').text,
                'organizacion': emisora.find('organizacion').text,
                'location': emisora.find('location').text,
                'concesionario': emisora.find('concesionario').text,
                'codigo': emisora.find('codigo').text,
                'frecuencia': emisora.find('frecuencia').text,
                'distintivo': emisora.find('distintivo').text,
                'tecnologia': emisora.find('tecnologia').text,
                'clase_programa': emisora.find('clase_programa').text,
                'latlong': emisora.find('latlong').text,
                'departamento': emisora.find('departamento').text,
                'distance': distance
            }
            result.append(station_info)

        return result

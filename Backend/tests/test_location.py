"""Unit Tests for GPS Navigation, NMEA Parsing, Safe-Zone Geofencing, and Google Maps API."""

import pytest
from smartvision_backend.location import (
    GPSCoordinates,
    GoogleMapsClient,
    LocationService,
    NMEAGPSParser,
    SafeHomeZone,
    haversine_distance_meters,
)


def test_haversine_distance():
    # Hyderabad Banjara Hills to Jubilee Hills (~3.5km)
    lat1, lon1 = 17.4399, 78.4482
    lat2, lon2 = 17.4320, 78.4120
    dist = haversine_distance_meters(lat1, lon1, lat2, lon2)
    assert 3000 < dist < 4500

    # Distance to self should be 0
    assert haversine_distance_meters(lat1, lon1, lat1, lon1) == 0.0


def test_nmea_parser_gprmc():
    parser = NMEAGPSParser()
    # Sample GPRMC sentence: 17 deg 26.394 min N, 78 deg 26.892 min E, 2.5 knots
    sentence = "$GPRMC,081836,A,1726.3940,N,07826.8920,E,002.5,054.7,241026,000.0,W*70"
    coords = parser.parse_sentence(sentence)
    assert coords is not None
    assert round(coords.latitude, 2) == 17.44
    assert round(coords.longitude, 2) == 78.45
    assert coords.speed_kmh > 4.0


def test_nmea_parser_gpgga():
    parser = NMEAGPSParser()
    sentence = "$GPGGA,081836,1726.3940,N,07826.8920,E,1,08,0.9,545.4,M,46.9,M,,*47"
    coords = parser.parse_sentence(sentence)
    assert coords is not None
    assert coords.satellites == 8
    assert coords.altitude_m == 545.4


def test_google_maps_urls():
    client = GoogleMapsClient()
    pin_url = client.get_google_maps_pin_url(17.4399, 78.4482)
    assert "https://www.google.com/maps?q=17.439900,78.448200" == pin_url

    directions_url = client.get_directions_url(17.4399, 78.4482)
    assert "https://www.google.com/maps/dir/?api=1&destination=17.439900,78.448200" == directions_url


def test_geofencing_safe_vs_wandering():
    home = SafeHomeZone(center_latitude=17.4399, center_longitude=78.4482, radius_meters=200.0)
    service = LocationService(home_zone=home)

    # 1. Update position 50m away from home -> inside safe zone
    st1 = service.update_gps_position(17.4402, 78.4484)
    assert st1["is_inside_safe_zone"] is True
    assert st1["wandering_alert_triggered"] is False

    # 2. Update position 600m away -> outside safe zone, triggers wandering alert
    st2 = service.update_gps_position(17.4450, 78.4530)
    assert st2["is_inside_safe_zone"] is False
    assert st2["wandering_alert_triggered"] is True
    assert "google_maps_url" in st2


def test_nearby_pharmacies_voice_response():
    service = LocationService()
    res = service.find_pharmacies_voice_response(language="te")
    assert "closest_pharmacy" in res
    assert len(res["pharmacies"]) >= 1
    assert "మీటర్ల దూరంలో ఉంది" in res["spoken_te"]

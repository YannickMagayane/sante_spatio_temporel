"""
Geographic reference data for all 26 provinces of the DRC.

Contains province capitals coordinates, boundaries, and geocoding utilities
for precise placement of health alerts on the interactive map.

Example:
    >>> from scripts.geo_reference import PROVINCES, geocode_location
    >>> PROVINCES["Nord-Kivu"]["capital"]
    'Goma'
    >>> geocode_location("Goma")
    {'lat': -1.6585, 'lon': 29.2200, 'province': 'Nord-Kivu'}
"""

# All 26 provinces of DRC with capital coordinates
PROVINCES = {
    "Kinshasa": {
        "capital": "Kinshasa",
        "lat": -4.3250, "lon": 15.3222,
        "bbox": [-4.60, 15.10, -4.10, 15.60],
    },
    "Kongo-Central": {
        "capital": "Matadi",
        "lat": -5.8167, "lon": 13.4500,
        "bbox": [-6.00, 12.20, -4.30, 16.20],
    },
    "Kwango": {
        "capital": "Kenge",
        "lat": -4.8100, "lon": 17.0400,
        "bbox": [-6.30, 16.30, -3.80, 18.60],
    },
    "Kwilu": {
        "capital": "Kikwit",
        "lat": -5.0366, "lon": 18.8162,
        "bbox": [-5.80, 18.00, -3.50, 20.00],
    },
    "Mai-Ndombe": {
        "capital": "Inongo",
        "lat": -1.9470, "lon": 18.2860,
        "bbox": [-3.50, 16.50, -0.50, 19.50],
    },
    "Équateur": {
        "capital": "Mbandaka",
        "lat": 0.0478, "lon": 18.2558,
        "bbox": [-1.50, 17.00, 2.00, 20.50],
    },
    "Mongala": {
        "capital": "Lisala",
        "lat": 2.1500, "lon": 21.5167,
        "bbox": [0.80, 20.00, 3.00, 23.00],
    },
    "Nord-Ubangi": {
        "capital": "Gbadolite",
        "lat": 4.2750, "lon": 21.0000,
        "bbox": [2.80, 19.50, 4.50, 22.50],
    },
    "Sud-Ubangi": {
        "capital": "Gemena",
        "lat": 3.2600, "lon": 19.7700,
        "bbox": [1.50, 18.00, 4.30, 21.00],
    },
    "Tshuapa": {
        "capital": "Boende",
        "lat": -0.2819, "lon": 20.8806,
        "bbox": [-2.00, 19.50, 0.80, 23.00],
    },
    "Tshopo": {
        "capital": "Kisangani",
        "lat": 0.5153, "lon": 25.1950,
        "bbox": [-1.00, 23.00, 2.50, 27.50],
    },
    "Bas-Uélé": {
        "capital": "Buta",
        "lat": 2.7833, "lon": 24.7333,
        "bbox": [1.50, 22.00, 4.00, 27.00],
    },
    "Haut-Uélé": {
        "capital": "Isiro",
        "lat": 2.7744, "lon": 27.6164,
        "bbox": [1.50, 26.00, 4.50, 30.50],
    },
    "Ituri": {
        "capital": "Bunia",
        "lat": 1.5594, "lon": 30.2522,
        "bbox": [0.50, 29.00, 3.00, 31.50],
    },
    "Nord-Kivu": {
        "capital": "Goma",
        "lat": -1.6585, "lon": 29.2200,
        "bbox": [-2.50, 27.50, 0.50, 29.80],
    },
    "Sud-Kivu": {
        "capital": "Bukavu",
        "lat": -2.5083, "lon": 28.8608,
        "bbox": [-4.50, 26.50, -1.50, 29.50],
    },
    "Maniema": {
        "capital": "Kindu",
        "lat": -2.9500, "lon": 25.9500,
        "bbox": [-4.50, 24.00, -1.00, 28.00],
    },
    "Haut-Katanga": {
        "capital": "Lubumbashi",
        "lat": -11.6876, "lon": 27.5026,
        "bbox": [-12.50, 25.50, -9.00, 30.00],
    },
    "Haut-Lomami": {
        "capital": "Kamina",
        "lat": -8.7353, "lon": 25.0000,
        "bbox": [-10.00, 23.50, -7.00, 27.00],
    },
    "Lualaba": {
        "capital": "Kolwezi",
        "lat": -10.7167, "lon": 25.4667,
        "bbox": [-12.00, 23.50, -9.00, 27.00],
    },
    "Tanganyika": {
        "capital": "Kalemie",
        "lat": -5.9333, "lon": 29.1833,
        "bbox": [-8.00, 26.00, -4.50, 30.00],
    },
    "Kasaï": {
        "capital": "Tshikapa",
        "lat": -6.4156, "lon": 20.7961,
        "bbox": [-7.50, 19.50, -4.50, 22.00],
    },
    "Kasaï-Central": {
        "capital": "Kananga",
        "lat": -5.8962, "lon": 22.4166,
        "bbox": [-7.00, 21.00, -4.00, 24.00],
    },
    "Kasaï-Oriental": {
        "capital": "Mbuji-Mayi",
        "lat": -6.1500, "lon": 23.6000,
        "bbox": [-7.50, 22.50, -4.50, 25.00],
    },
    "Lomami": {
        "capital": "Kabinda",
        "lat": -6.1333, "lon": 24.4833,
        "bbox": [-7.50, 23.00, -4.00, 26.00],
    },
    "Sankuru": {
        "capital": "Lusambo",
        "lat": -4.9667, "lon": 23.4500,
        "bbox": [-5.50, 22.00, -2.50, 25.00],
    },
}

# Known cities and towns with coordinates for geocoding
KNOWN_LOCATIONS = {}

# Populate from province capitals
for prov_name, prov_data in PROVINCES.items():
    capital = prov_data["capital"]
    KNOWN_LOCATIONS[capital.lower()] = {
        "lat": prov_data["lat"],
        "lon": prov_data["lon"],
        "province": prov_name,
        "name": capital,
    }
    # Also add the province name itself
    KNOWN_LOCATIONS[prov_name.lower()] = {
        "lat": prov_data["lat"],
        "lon": prov_data["lon"],
        "province": prov_name,
        "name": capital,
    }

# Additional known cities/towns beyond province capitals
_EXTRA_LOCATIONS = {
    "beni":       {"lat": 0.4950, "lon": 29.4625, "province": "Nord-Kivu"},
    "butembo":    {"lat": 0.1417, "lon": 29.2833, "province": "Nord-Kivu"},
    "kipushi":    {"lat": -11.7614, "lon": 27.2510, "province": "Haut-Katanga"},
    "likasi":     {"lat": -10.9833, "lon": 26.7333, "province": "Haut-Katanga"},
    "mbuji-mayi": {"lat": -6.1500, "lon": 23.6000, "province": "Kasaï-Oriental"},
    "uvira":      {"lat": -3.3936, "lon": 29.1378, "province": "Sud-Kivu"},
    "kalemie":    {"lat": -5.9333, "lon": 29.1833, "province": "Tanganyika"},
    "bandundu":   {"lat": -3.3175, "lon": 17.3806, "province": "Kwilu"},
    "mbanza-ngungu": {"lat": -5.2500, "lon": 14.8667, "province": "Kongo-Central"},
    "boma":       {"lat": -5.8514, "lon": 13.0500, "province": "Kongo-Central"},
    "kananga":    {"lat": -5.8962, "lon": 22.4166, "province": "Kasaï-Central"},
    "kinshasa":   {"lat": -4.3250, "lon": 15.3222, "province": "Kinshasa"},
    "lubumbashi": {"lat": -11.6876, "lon": 27.5026, "province": "Haut-Katanga"},
    "goma":       {"lat": -1.6585, "lon": 29.2200, "province": "Nord-Kivu"},
    "bukavu":     {"lat": -2.5083, "lon": 28.8608, "province": "Sud-Kivu"},
    "kisangani":  {"lat": 0.5153, "lon": 25.1950, "province": "Tshopo"},
    "kolwezi":    {"lat": -10.7167, "lon": 25.4667, "province": "Lualaba"},
    "tshikapa":   {"lat": -6.4156, "lon": 20.7961, "province": "Kasaï"},
    "kamina":     {"lat": -8.7353, "lon": 25.0000, "province": "Haut-Lomami"},
    "mbandaka":   {"lat": 0.0478, "lon": 18.2558, "province": "Équateur"},
    "lisala":     {"lat": 2.1500, "lon": 21.5167, "province": "Mongala"},
    "gemena":     {"lat": 3.2600, "lon": 19.7700, "province": "Sud-Ubangi"},
    "gbadolite":  {"lat": 4.2750, "lon": 21.0000, "province": "Nord-Ubangi"},
    "boende":     {"lat": -0.2819, "lon": 20.8806, "province": "Tshuapa"},
    "isiro":      {"lat": 2.7744, "lon": 27.6164, "province": "Haut-Uélé"},
    "buta":       {"lat": 2.7833, "lon": 24.7333, "province": "Bas-Uélé"},
    "inongo":     {"lat": -1.9470, "lon": 18.2860, "province": "Mai-Ndombe"},
    "matadi":     {"lat": -5.8167, "lon": 13.4500, "province": "Kongo-Central"},
    "kenge":      {"lat": -4.8100, "lon": 17.0400, "province": "Kwango"},
    "kikwit":     {"lat": -5.0366, "lon": 18.8162, "province": "Kwilu"},
    "kindu":      {"lat": -2.9500, "lon": 25.9500, "province": "Maniema"},
    "kabinda":    {"lat": -6.1333, "lon": 24.4833, "province": "Lomami"},
    "lusambo":    {"lat": -4.9667, "lon": 23.4500, "province": "Sankuru"},
    "fungurume":  {"lat": -10.6167, "lon": 26.3000, "province": "Lualaba"},
    "karisimbi":  {"lat": -1.5000, "lon": 29.3833, "province": "Nord-Kivu"},
    "rutshuru":   {"lat": -1.1833, "lon": 29.4500, "province": "Nord-Kivu"},
    "masisi":     {"lat": -1.4000, "lon": 28.8167, "province": "Nord-Kivu"},
    "walikale":   {"lat": -1.4333, "lon": 28.0667, "province": "Nord-Kivu"},
    "nyiragongo": {"lat": -1.5209, "lon": 29.2500, "province": "Nord-Kivu"},
    "minova":     {"lat": -1.8833, "lon": 28.9333, "province": "Sud-Kivu"},
    "fizi":       {"lat": -4.3000, "lon": 28.9333, "province": "Sud-Kivu"},
    "baraka":     {"lat": -4.1000, "lon": 29.1000, "province": "Sud-Kivu"},
    "aru":        {"lat": 2.8500, "lon": 30.8333, "province": "Ituri"},
    "djugu":      {"lat": 1.9167, "lon": 30.5000, "province": "Ituri"},
    "mambasa":    {"lat": 1.3833, "lon": 29.0500, "province": "Ituri"},
    "mahagi":     {"lat": 2.3167, "lon": 30.9833, "province": "Ituri"},
}

for loc_name, loc_data in _EXTRA_LOCATIONS.items():
    KNOWN_LOCATIONS[loc_name] = {
        "lat": loc_data["lat"],
        "lon": loc_data["lon"],
        "province": loc_data["province"],
        "name": loc_name.capitalize(),
    }


def geocode_location(text):
    """
    Extract and geocode a location from a text string.

    Searches for known DRC city/province names in the text and returns
    the coordinates of the first match found.

    :param text: Text containing a potential location name.
    :return: Dict with lat, lon, province, name or None if not found.

    Example:
        >>> geocode_location("Épidémie de choléra signalée à Goma")
        {'lat': -1.6585, 'lon': 29.22, 'province': 'Nord-Kivu', 'name': 'Goma'}

        >>> geocode_location("Aucun lieu mentionné") is None
        True
    """
    text_lower = text.lower()

    # Try exact city matches first (longer names first to avoid partial matches)
    sorted_locations = sorted(KNOWN_LOCATIONS.keys(), key=len, reverse=True)

    for loc_name in sorted_locations:
        if loc_name in text_lower:
            return KNOWN_LOCATIONS[loc_name].copy()

    return None


def geocode_or_default(text, default_city="kinshasa"):
    """
    Geocode from text, falling back to a default city.

    :param text: Text to search for locations.
    :param default_city: Fallback city name.
    :return: Dict with lat, lon, province, name.

    Example:
        >>> result = geocode_or_default("Texte sans lieu")
        >>> result["name"]
        'Kinshasa'
    """
    result = geocode_location(text)
    if result:
        return result
    return KNOWN_LOCATIONS.get(default_city.lower(), KNOWN_LOCATIONS["kinshasa"]).copy()


def get_all_province_coords():
    """
    Return a dict of all provinces with their coordinates.

    :return: Dict keyed by province name.

    Example:
        >>> coords = get_all_province_coords()
        >>> len(coords) == 26
        True
    """
    return {name: {"lat": p["lat"], "lon": p["lon"], "capital": p["capital"]}
            for name, p in PROVINCES.items()}

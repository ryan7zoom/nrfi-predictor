"""
Static ballpark data: coordinates, CF orientation, roof status, park factors.
Keyed by MLB team_id. Coordinates/orientation verified during research phase;
park factors from published aggregate data.
"""

from dataclasses import dataclass


TEAM_NAMES = {
    108: "Los Angeles Angels", 109: "Arizona Diamondbacks", 110: "Baltimore Orioles",
    111: "Boston Red Sox", 112: "Chicago Cubs", 113: "Cincinnati Reds",
    114: "Cleveland Guardians", 115: "Colorado Rockies", 116: "Detroit Tigers",
    117: "Houston Astros", 118: "Kansas City Royals", 119: "Los Angeles Dodgers",
    120: "Washington Nationals", 121: "New York Mets", 133: "Athletics",
    134: "Pittsburgh Pirates", 135: "San Diego Padres", 136: "Seattle Mariners",
    137: "San Francisco Giants", 138: "St. Louis Cardinals", 139: "Tampa Bay Rays",
    140: "Texas Rangers", 141: "Toronto Blue Jays", 142: "Minnesota Twins",
    143: "Philadelphia Phillies", 144: "Atlanta Braves", 145: "Chicago White Sox",
    146: "Miami Marlins", 147: "New York Yankees", 158: "Milwaukee Brewers",
}


# Verified via GET /api/v1/teams/{teamId}?hydrate=venue(location)
PARK_COORDINATES = {
    108: {"venue": "Angel Stadium", "lat": 33.80019044, "lon": -117.8823996},
    109: {"venue": "Chase Field", "lat": 33.445302, "lon": -112.066687},
    110: {"venue": "Oriole Park at Camden Yards", "lat": 39.283787, "lon": -76.621689},
    111: {"venue": "Fenway Park", "lat": 42.346456, "lon": -71.097441},
    112: {"venue": "Wrigley Field", "lat": 41.948171, "lon": -87.655503},
    113: {"venue": "Great American Ball Park", "lat": 39.097389, "lon": -84.506611},
    114: {"venue": "Progressive Field", "lat": 41.495861, "lon": -81.685255},
    115: {"venue": "Coors Field", "lat": 39.756042, "lon": -104.994136},
    116: {"venue": "Comerica Park", "lat": 42.3391151, "lon": -83.048695},
    117: {"venue": "Daikin Park", "lat": 29.756967, "lon": -95.355509},
    118: {"venue": "Kauffman Stadium", "lat": 39.051567, "lon": -94.480483},
    119: {"venue": "Dodger Stadium", "lat": 34.07368, "lon": -118.24053},
    120: {"venue": "Nationals Park", "lat": 38.872861, "lon": -77.007501},
    121: {"venue": "Citi Field", "lat": 40.75753012, "lon": -73.84559155},
    133: {"venue": "Sutter Health Park", "lat": 38.57994, "lon": -121.51246},
    134: {"venue": "PNC Park", "lat": 40.446904, "lon": -80.005753},
    135: {"venue": "Petco Park", "lat": 32.707861, "lon": -117.157278},
    136: {"venue": "T-Mobile Park", "lat": 47.591333, "lon": -122.33251},
    137: {"venue": "Oracle Park", "lat": 37.778383, "lon": -122.389448},
    138: {"venue": "Busch Stadium", "lat": 38.62256667, "lon": -90.19286667},
    139: {"venue": "Tropicana Field", "lat": 27.767778, "lon": -82.6525},
    140: {"venue": "Globe Life Field", "lat": 32.747299, "lon": -97.081818},
    141: {"venue": "Rogers Centre", "lat": 43.64155, "lon": -79.38915},
    142: {"venue": "Target Field", "lat": 44.981829, "lon": -93.277891},
    143: {"venue": "Citizens Bank Park", "lat": 39.90539086, "lon": -75.16716957},
    144: {"venue": "Truist Park", "lat": 33.890672, "lon": -84.467641},
    145: {"venue": "Rate Field", "lat": 41.83, "lon": -87.634167},
    146: {"venue": "loanDepot park", "lat": 25.77796236, "lon": -80.21951795},
    147: {"venue": "Yankee Stadium", "lat": 40.82919482, "lon": -73.9264977},
    158: {"venue": "American Family Field", "lat": 43.02838, "lon": -87.97099},
}


# Source: Clem's Baseball (Lowry's Green Cathedrals, Ritter, ESPN Almanac)
PARK_CF_ORIENTATION = {
    108: "NE", 109: "N", 110: "NNE", 111: "NE", 112: "NE", 113: "ESE",
    114: "N", 115: "N", 116: "SSE", 117: "ENE", 118: "NE", 119: "NNE",
    120: "NNE", 121: "NNE", 133: "NE", 134: "ESE", 135: "N", 136: "NE",
    137: "ESE", 138: "NE", 139: "NE", 140: "ENE", 141: "NNW", 142: "E",
    143: "NNE", 144: "SSE", 145: "ESE", 146: "ESE", 147: "ENE", 158: "SE",
}

COMPASS_TO_DEG = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


# Confirmed: no live per-game roof status field exists in the API.
# roofType is a static per-venue property, not per-game. Zero wind effect
# unconditionally for these teams.
ROOFED_TEAM_IDS = {109, 140, 117, 158, 146, 136, 141, 139}

# Documented non-geometric wind behavior. Not modeled with a correction
# term yet, flagged for caution / wider tolerance.
IRREGULAR_WIND_TEAM_IDS = {112, 137}


# Run factor, 100 = neutral. Extreme parks included intentionally.
PARK_FACTORS_RUNS = {
    108: 97, 109: 103, 110: 100, 111: 104, 112: 102, 113: 106, 114: 96,
    115: 112, 116: 99, 117: 98, 118: 99, 119: 97, 120: 98, 121: 95,
    133: 99, 134: 96, 135: 94, 136: 93, 137: 92, 138: 98, 139: 96,
    140: 101, 141: 101, 142: 100, 143: 102, 144: 100, 145: 103,
    146: 94, 147: 102, 158: 99,
}


def get_park_factor(team_id: int) -> int:
    return PARK_FACTORS_RUNS[team_id]


def get_coordinates(team_id: int) -> dict:
    return PARK_COORDINATES[team_id]


def get_cf_orientation_degrees(team_id: int) -> float:
    return COMPASS_TO_DEG[PARK_CF_ORIENTATION[team_id]]


def get_blowout_wind_direction_degrees(team_id: int) -> float:
    return (get_cf_orientation_degrees(team_id) + 180) % 360


def wind_effect(wind_from_deg: float, wind_speed: float, team_id: int, tolerance: float = 30) -> float:
    if team_id in ROOFED_TEAM_IDS:
        return 0.0

    out_deg = get_blowout_wind_direction_degrees(team_id)
    cf_deg = get_cf_orientation_degrees(team_id)

    diff_out = min(abs((wind_from_deg - out_deg) % 360), 360 - abs((wind_from_deg - out_deg) % 360))
    diff_in = min(abs((wind_from_deg - cf_deg) % 360), 360 - abs((wind_from_deg - cf_deg) % 360))

    if diff_out <= tolerance:
        return wind_speed
    elif diff_in <= tolerance:
        return -wind_speed
    return 0.0


@dataclass
class ParkInfo:
    team_id: int
    name: str
    venue: str
    lat: float
    lon: float
    park_factor: int
    cf_orientation_deg: float
    is_roofed: bool
    is_irregular_wind: bool


def get_park_info(team_id: int) -> ParkInfo:
    coords = get_coordinates(team_id)
    return ParkInfo(
        team_id=team_id,
        name=TEAM_NAMES[team_id],
        venue=coords["venue"],
        lat=coords["lat"],
        lon=coords["lon"],
        park_factor=get_park_factor(team_id),
        cf_orientation_deg=get_cf_orientation_degrees(team_id),
        is_roofed=team_id in ROOFED_TEAM_IDS,
        is_irregular_wind=team_id in IRREGULAR_WIND_TEAM_IDS,
    )

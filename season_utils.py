"""
Utility functions for season and month-group handling.

Supports both traditional season codes (DJF, MAM, JJA, SON) and custom
month groups specified as lists of month numbers (e.g., [8, 9, 10, 11]).

Custom month groups get a label built from the first letter of each month name,
e.g., [8, 9, 10, 11] -> "ASON".
"""

# Month number -> first letter of month name
MONTH_INITIALS = {
    1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J',
    7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'
}

# Standard season definitions
STANDARD_SEASONS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}


def resolve_season(season):
    """Resolve a season specification to (label, months_list).

    Parameters
    ----------
    season : str or list of int or None
        - A standard season code: "DJF", "MAM", "JJA", "SON"
        - A custom month-initial string: e.g. "ASON", "JJA", "JJAS"
        - A list of month numbers: e.g. [8, 9, 10, 11]
        - None: no season filtering

    Returns
    -------
    tuple (str or None, list of int or None)
        (label, months) where label is the display name and months is the
        list of month numbers, or (None, None) if no filtering.
    """
    if season is None:
        return None, None

    if isinstance(season, list):
        # List of month numbers, e.g. [8, 9, 10, 11]
        months = [int(m) for m in season]
        label = ''.join(MONTH_INITIALS[m] for m in months)
        return label, months

    if isinstance(season, str):
        # Check standard seasons first
        if season in STANDARD_SEASONS:
            return season, STANDARD_SEASONS[season]
        # Treat as a custom month-initial string, e.g. "ASON", "JJAS"
        # Build reverse mapping: initial -> possible months
        months = _initials_to_months(season)
        if months is not None:
            return season, months
        # Unknown string
        return season, None

    return None, None


def _initials_to_months(initials):
    """Convert a string of month initials to a list of month numbers.

    Resolves ambiguity (J=Jan/Jun/Jul, M=Mar/May, A=Apr/Aug) by finding
    a contiguous (wrapping) sequence of months whose initials match.

    Parameters
    ----------
    initials : str
        E.g. "ASON", "JJAS", "NDJF", "MAMJJA"

    Returns
    -------
    list of int or None
        List of month numbers, or None if the string can't be resolved.
    """
    initials = initials.upper()
    n = len(initials)
    if n == 0 or n > 12:
        return None

    # Try every possible starting month and check if n consecutive months
    # produce the requested initials string
    for start in range(1, 13):
        months = [(start + i - 1) % 12 + 1 for i in range(n)]
        candidate = ''.join(MONTH_INITIALS[m] for m in months)
        if candidate == initials:
            return months

    return None


def parse_seasons_config(seasons_config):
    """Parse the season config value into a list of season entries to process.

    Parameters
    ----------
    seasons_config : None, str, list
        The value from config['filter']['season'].
        Can be:
        - None: no season filtering
        - A string: single season code (e.g., "DJF" or "ASON")
        - A list that can contain:
          - strings: season codes (e.g., ["DJF", "MAM", "ASON"])
          - lists of ints: month groups (e.g., [[8,9,10,11], [12,1,2]])
          - mixed: ["DJF", [8,9,10,11]]

    Returns
    -------
    list
        List of season entries (each is str, list-of-int, or None).
    """
    if seasons_config is None:
        return [None]
    if isinstance(seasons_config, str):
        return [seasons_config]
    if isinstance(seasons_config, list):
        if not seasons_config:
            return [None]
        # Check if it's a flat list of ints (a single month group)
        if all(isinstance(x, int) for x in seasons_config):
            return [seasons_config]
        return seasons_config
    return [None]

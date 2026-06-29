"""
Shared utility functions
"""


def format_threshold_string(config):
    """
    Format threshold value for filename (e.g., '1st', '99th', '30mm')
    Returns empty string if threshold method not configured yet
    """
    if 'threshold' not in config:
        return ""

    threshold_cfg = config['threshold']
    method = threshold_cfg['method']
    variable = config['variable']

    if method == 'fixed':
        # For fixed values, use the value with appropriate unit
        threshold_value = threshold_cfg['fixed']['value']
        if variable == 'tp24':
            return f"{threshold_value:.0f}mm"
        elif variable == '2t':
            return f"{threshold_value:.1f}C"
        elif variable == '10ff':
            return f"{threshold_value:.1f}ms"
        else:
            return f"{threshold_value:.1f}"

    elif method in ['dataset_climatology', 'station_climatology']:
        # For percentile-based thresholds
        percentile = threshold_cfg[method]['percentile']

        # Format with proper ordinal suffix
        if percentile == 1:
            return "1st"
        elif percentile == 2:
            return "2nd"
        elif percentile == 3:
            return "3rd"
        elif percentile == 99:
            return "99th"
        else:
            return f"{percentile}th"

    return ""

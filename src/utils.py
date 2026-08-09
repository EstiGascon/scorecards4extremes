"""
Shared utility functions
"""


def compute_steps(config):
    """
    Derive the list of forecast lead-time steps (hours) a config needs.

    Single source of truth shared by extraction (extract_points.run_step3) and
    MARS retrieval (mars_retrieve), so the retrieved steps always match what the
    extractor will look for.

    Returns (steps, step_to_forecast_day):
      - steps: sorted list of unique step hours
      - step_to_forecast_day: {step: forecast_day} (empty when explicit steps are used)
    """
    step_to_forecast_day = {}

    if 'forecast_days' in config and config['forecast_days'] is not None:
        forecast_days = config['forecast_days']
        frequency = config.get('lead_time_frequency', 6)  # Default 6h

        steps = []
        for day in forecast_days:
            # Day 1 = 1-24h, Day 2 = 25-48h, etc. (matches: day = ((lt-1)//24)+1)
            day_start = (day - 1) * 24 + 1  # 1, 25, 49, 73, 97, ...
            day_end = day * 24 + 1          # 25, 49, 73, 97, 121, ... (exclusive)

            # Align start to the frequency grid (e.g. freq=6 -> 6, 12, 18, 24)
            if frequency > 1:
                offset = frequency - (day_start % frequency) if (day_start % frequency) != 0 else 0
                day_start += offset
            day_steps = list(range(day_start, day_end, frequency))

            for step in day_steps:
                if step not in steps:
                    steps.append(step)
                    step_to_forecast_day[step] = day

        # Remove step 0 for precipitation variables (need 24h accumulation)
        if config['variable'] in ['tp24', 'tp'] and 0 in steps:
            steps.remove(0)
            step_to_forecast_day.pop(0, None)

        steps = sorted(steps)
    else:
        # Explicit steps
        steps = list(config['steps'])

    return steps, step_to_forecast_day


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



def convert_to_seconds(val, t):
    """
    Convert a given unit of time to seconds
    Args:
        val (int): Value in number of years/days/minutes
        t (str): Specifies the type of value. It can be one of the following
                'y': Val denotes years
                'd': Val denotes days
                'h': Val denotes hours
                'm': Val denotes minutes
                's': Val denotes seconds

    Returns: Time in seconds
    """
    result = -1
    if t == 'y':
        result = val*365*24*3600
    elif t == 'd':
        result = val*24*3600
    elif t == 'h':
        result = val*3600
    elif t == 'm':
        result = val*60
    elif t == 's':
        result = val

    return result

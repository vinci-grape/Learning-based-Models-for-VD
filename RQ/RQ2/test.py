def hex_to_rgb(hex_color):
    """
    Convert a hexadecimal color representation to RGB.

    Parameters:
        hex_color (str): Hexadecimal color string (e.g., "#RRGGBB" or "RRGGBB").

    Returns:
        tuple: RGB tuple representing the color (R, G, B).
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    """
    Convert an RGB color representation to hexadecimal.

    Parameters:
        rgb_color (tuple): RGB tuple representing the color (R, G, B).

    Returns:
        str: Hexadecimal color string (e.g., "#RRGGBB").
    """
    return f"#{rgb_color[0]:02X}{rgb_color[1]:02X}{rgb_color[2]:02X}"


def interpolate_color(color1, color2 , cur_val ):
    """
    Interpolate between two hexadecimal colors.

    Parameters:
        color1 (str): Hexadecimal color string (e.g., "#RRGGBB" or "RRGGBB").
        color2 (str): Hexadecimal color string (e.g., "#RRGGBB" or "RRGGBB").
        steps (int): Number of steps for interpolation.

    Returns:
        list: List of interpolated hexadecimal colors.
    """
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2

    # Calculate the step size for each channel
    r= r1 + (r2 - r1) * cur_val
    g= g1 + (g2 - g1) * cur_val
    b= b1 + (b2 - b1) * cur_val


    return rgb_to_hex((int(r), int(g), int(b)))

# Example usage:
color1_hex = "#FFFFFF"  # Red
color2_hex = "#067005"  # Blue


data = [0.2,0.000]
for item in data:
    print(interpolate_color(color1_hex, color2_hex, item))


import numpy as np


def normalize_angle(angle):
    normalized_angle = abs(angle) % (2 * np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2 * np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle

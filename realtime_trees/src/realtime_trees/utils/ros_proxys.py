from typing import Tuple

import numpy as np


class Time:
    """Simple proxy for the time object in ROS consisting of seconds and nanoseconds
       Adapted from https://github.com/ros2/rclpy/blob/d387309e304fbdf73e8b54e6f1f1c6bfaa593545/rclpy/rclpy/time.py
    """
    CONVERSION_CONSTANT = 10 ** 9

    def __init__(self, seconds: int, nanoseconds: int) -> None:
        if seconds < 0:
            raise ValueError('Seconds value must not be negative')
        if nanoseconds < 0:
            raise ValueError('Nanoseconds value must not be negative')
        total_nanoseconds = int(seconds * Time.CONVERSION_CONSTANT)
        total_nanoseconds += int(nanoseconds)
        if total_nanoseconds >= 2**63:
            raise OverflowError(
                'Total nanoseconds value is too large to store in C time point.')
        self._nanoseconds = total_nanoseconds

    @property
    def nanoseconds(self):
        """
        Get the time (seconds + nanoseconds) as a single number in nanoseconds
        """
        return self._nanoseconds

    @property
    def secs(self):
        """
        return the seconds only
        """
        seconds, nanoseconds = self.seconds_nanoseconds()
        return seconds
    
    @property
    def nsecs(self):
        """
        return the nanoseconds only
        """
        seconds, nanoseconds = self.seconds_nanoseconds()
        return nanoseconds

    def seconds_nanoseconds(self) -> Tuple[int, int]:
        """
        Get time as separate seconds and nanoseconds components.
        """
        nanoseconds = self._nanoseconds
        return (nanoseconds // Time.CONVERSION_CONSTANT, nanoseconds % Time.CONVERSION_CONSTANT)

    def to_msg(self):
        """
        Convert the time to a ROS 2 message
        """
        import builtin_interfaces.msg
        seconds, nanoseconds = self.seconds_nanoseconds()
        return builtin_interfaces.msg.Time(sec=seconds, nanosec=nanoseconds)

    @classmethod
    def from_msg(cls, msg):
        """
        Instantiate time from ROS 2 message
        """
        import builtin_interfaces.msg
        if not isinstance(msg, builtin_interfaces.msg.Time):
            raise TypeError('Must pass a builtin_interfaces.msg.Time object')
        return cls(seconds=msg.sec, nanoseconds=msg.nanosec)

    def __repr__(self) -> str:
        seconds, nanoseconds = self.seconds_nanoseconds()
        return f"Time(secs={seconds}, nsecs={nanoseconds:0>9})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Time):
            return self.nanoseconds == other.nanoseconds
        raise TypeError("Can't compare time with object of type: ", type(other))
    
    def __lt__(self, other: object) -> bool:
        if isinstance(other, Time):
            return self.nanoseconds < other.nanoseconds
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Time):
            return self.nanoseconds > other.nanoseconds
        return NotImplemented
    
    def __hash__(self) -> int:
        return hash(self.nanoseconds)
    
    def __sub__(self, other: object) -> float:
        if isinstance(other, Time):
            # Probably it would be best to return a duration in order to be consistent
            return self.nanoseconds - other.nanoseconds
        return NotImplemented

class StampedPose:
    """Simple proxy for the stamped pose object in ROS consisting of a pose and a Time
    """
    def __init__(self, pose: np.ndarray, time_stamp: Time) -> None:
        self.pose = pose
        self.time_stamp = time_stamp
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, StampedPose):
            return np.allclose(self.pose, other.pose) and self.time_stamp == other.time_stamp
        return False
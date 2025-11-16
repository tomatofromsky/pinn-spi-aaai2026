import json
import time
from typing import Union, Optional

class MetricLogger:
    """JSONL logger for experiment metrics with timestamps and timing support"""
    def __init__(self, path: str):
        self.path = path
        self.fp = open(path, "a", encoding="utf-8")
        self.start_time = time.time()

        # Training time tracking (excludes evaluation time)
        self.training_start_time = None
        self.total_training_time = 0.0

    def log_scalar(self, name: str, value: Union[float, int], step: int,
                   wall_time: Optional[float] = None, training_time: Optional[float] = None):
        """
        Log a scalar metric with timestamp information

        Args:
            name: Metric name (e.g., 'eval/avg_return')
            value: Metric value
            step: Training step/iteration number
            wall_time: Wall clock time since experiment start (auto-computed if None)
            training_time: Cumulative training time excluding evaluation (auto-tracked)
        """
        current_time = time.time()
        record = {
            "step": step,
            "name": name,
            "value": float(value),
            "timestamp": current_time,
            "wall_time": wall_time if wall_time is not None else (current_time - self.start_time),
            "training_time": training_time if training_time is not None else self.total_training_time
        }
        self.fp.write(json.dumps(record) + "\n")
        self.fp.flush()

    def start_training_timer(self):
        """Start timing a training phase (excludes evaluation)"""
        self.training_start_time = time.time()

    def end_training_timer(self):
        """End timing a training phase and accumulate the duration"""
        if self.training_start_time is not None:
            duration = time.time() - self.training_start_time
            self.total_training_time += duration
            self.training_start_time = None
            return duration
        return 0.0

    def get_training_time(self) -> float:
        """Get total accumulated training time"""
        return self.total_training_time

    def get_wall_time(self) -> float:
        """Get total wall clock time since experiment start"""
        return time.time() - self.start_time

    def flush(self):
        """Flush the file buffer"""
        self.fp.flush()

    def close(self):
        """Close the logger"""
        self.fp.close()
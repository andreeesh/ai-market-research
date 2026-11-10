from dataclasses import dataclass


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    timeout: float = 1.0
    jitter: float = 0.5

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts should be >= 1")

        if self.timeout <= 0:
            raise ValueError("timeout should be > 0")

        if self.jitter < 0:
            raise ValueError("jitter should be >= 0")

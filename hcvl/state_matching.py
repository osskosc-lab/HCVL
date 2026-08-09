def matched(x_ab: float, x_ba: float, epsilon: float = 0.75) -> bool:
    return abs(float(x_ab)-float(x_ba)) < float(epsilon)

def retention_rate(retained: int, total: int) -> float:
    return float(retained)/float(total) if total else 0.0

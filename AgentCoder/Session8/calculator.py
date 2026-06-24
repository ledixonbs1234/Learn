

# calculator.py
def divide(a: int, b: int) -> float:
    # Xử lý trường hợp chia cho 0
    if b == 0:
        return 0.0
    return a / b


def add(a: int, b: int) -> int:
    return a + b
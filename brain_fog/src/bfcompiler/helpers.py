from typing import List, Set, Tuple


def factor_number(target: int) -> Tuple[int, int]:
    """Returns the two closest factors to each other that multiply to the input number.

    Args:
        target (int): The number to factor.

    Returns:
        Tuple[int, int]: A tuple of the two numbers.
    """
    nums: Set[int] = set(range(1, target + 1))
    pairs: List[Tuple[int, int]] = [
        (n, target // n) for n in nums if target / n in nums
    ]

    # remove duplicates
    new_pairs: List[Tuple[int, int]] = []
    for pair in pairs:
        if pair[0] >= pair[1]:
            new_pairs.append((pair[1], pair[0]))

    current_best: Tuple[int, int] | None = None
    current_best_difference: int | None = None

    for pair in new_pairs:
        if current_best is None:
            current_best = pair
            current_best_difference = pair[1] - pair[0]
            continue

        if (pair[1] - pair[0]) < current_best_difference:
            current_best = pair
            current_best_difference = pair[1] - pair[0]

    return current_best


if __name__ == "__main__":
    print(factor_number(7))
    print(factor_number(25))
    print(factor_number(60))
    print(factor_number(5908))
    print(factor_number(905844))

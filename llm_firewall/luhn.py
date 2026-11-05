"""The Luhn (mod 10) checksum, used by credit-card numbers and Canadian SINs.

Algorithm, from the RIGHTMOST digit moving left:
  1. Double every second digit (the 2nd, 4th, 6th ... from the right).
  2. If doubling gives a two-digit number, subtract 9 (same as adding its digits).
  3. Sum everything. The number is valid iff the sum is a multiple of 10.

Example: SIN 046 454 286
  digits         0 4 6 4 5 4 2 8 6
  double 2nd,4th from right:   positions (from right) 2,4,6,8 -> 8, 4, 4, 4
  0 + (4*2=8) + 6 + (4*2=8) + 5 + (4*2=8) + 2 + (8*2=16-9=7) + 6 = 50  -> valid

Luhn catches every single-digit typo and most swaps of adjacent digits. For us
it is a cheap filter: a random 16-digit number passes only 1 time in 10, so
requiring Luhn removes ~90% of false "credit card" matches (order numbers,
phone numbers, timestamps ...).
"""

from __future__ import annotations


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 2:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:  # every second digit from the right
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

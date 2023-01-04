def func(n):
    if n == 0:
        return 2
    if n == 1:
        return 1
    return func(n-1) + func(n-2)


if __name__ == "__main__":
    res = func(32)
    print(res)
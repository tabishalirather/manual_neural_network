def inputs():
    inputs = [
        [0, 0, 0, 0],  # n=0
        [0, 0, 0, 1],  # n=1
        [0, 0, 1, 0],  # n=2
        [0, 0, 1, 1],  # n=3
        [0, 1, 0, 0],  # n=4
        [0, 1, 0, 1],  # n=5
        [0, 1, 1, 0],  # n=6
        [0, 1, 1, 1],  # n=7
        [1, 0, 0, 0],  # n=8
        [1, 0, 0, 1],  # n=9
    ]
    return inputs


def labels():

    labels = [
        [1, 0],  # n=0 even
        [0, 1],  # n=1 odd
        [1, 0],  # n=2 even
        [0, 1],  # n=3 odd
        [1, 0],  # n=4 even
        [0, 1],  # n=5 odd
        [1, 0],  # n=6 even
        [0, 1],  # n=7 odd
        [1, 0],  # n=8 even
        [0, 1],  # n=9 odd
    ]
    return labels

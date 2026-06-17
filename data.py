import torch


def make_bars_stripes(n=4, dtype=torch.float32):
    """
    Generate unique n x n Bars & Stripes binary patterns.

    Bars:
        each row is either all 0 or all 1.

    Stripes:
        each column is either all 0 or all 1.

    For n=4, this gives:
        2^4 + 2^4 - 2 = 30 unique patterns.

    Returns:
        x: tensor of shape (num_patterns, n*n), entries 0 or 1.
    """
    patterns = []

    # Horizontal bars.
    for mask in range(2**n):
        img = torch.zeros(n, n, dtype=dtype)
        for row in range(n):
            bit = (mask >> row) & 1
            img[row, :] = float(bit)
        patterns.append(img.flatten())

    # Vertical stripes.
    for mask in range(2**n):
        img = torch.zeros(n, n, dtype=dtype)
        for col in range(n):
            bit = (mask >> col) & 1
            img[:, col] = float(bit)
        patterns.append(img.flatten())

    x = torch.stack(patterns, dim=0)

    # Remove duplicates: all-zero and all-one appear in both sets.
    x = torch.unique(x, dim=0)

    return x


def pixels_to_voltages(x, low=-0.5, high=0.5):
    """
    Map binary pixel values 0/1 to input voltages.

        0 -> low
        1 -> high

    We start with a conservative common voltage range [-0.5, 0.5],
    which lies safely inside the allowed ranges of the data electrodes.
    """
    return low + x * (high - low)


def voltages_to_pixels(v, low=-0.5, high=0.5):
    """
    Inverse affine map from voltages back to pixel units.
    """
    return (v - low) / (high - low)


def print_patterns(x, n=4, max_patterns=5):
    """
    Small debugging helper.
    """
    count = min(max_patterns, x.shape[0])
    for i in range(count):
        print(f"\nPattern {i}:")
        print(x[i].reshape(n, n))


if __name__ == "__main__":
    x = make_bars_stripes(n=4)

    print("Bars & Stripes dataset")
    print("  shape:", tuple(x.shape))
    print("  number of patterns:", x.shape[0])
    print("  unique values:", torch.unique(x).tolist())

    xv = pixels_to_voltages(x)
    print("\nVoltage encoding")
    print("  shape:", tuple(xv.shape))
    print("  min:", xv.min().item())
    print("  max:", xv.max().item())

    x_back = voltages_to_pixels(xv)
    print("\nRound-trip check")
    print("  max abs error:", (x_back - x).abs().max().item())

    print_patterns(x, n=4, max_patterns=4)

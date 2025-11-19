import yaml
import numpy as np

ADJ = [
    [0, 1, 2, 4, 5, 8, 9, 22],
    [0, 1, 2, 3, 4, 8, 22],
    [0, 1, 2, 3, 22],
    [1, 2, 3],
    [0, 1, 4, 5, 6, 8, 9, 22],
    [0, 4, 5, 6, 7, 8, 22],
    [4, 5, 6, 7],
    [5, 6, 7],
    [0, 1, 4, 5, 8, 9, 10, 22],
    [0, 4, 8, 9, 10, 11, 22],
    [8, 9, 10, 11, 12, 14, 18],
    [9, 10, 11, 12, 13, 14, 15, 18, 19],
    [10, 11, 12, 13, 14, 15, 18, 19],
    [11, 12, 13, 14, 18],
    [10, 11, 12, 13, 14, 15, 16, 18, 19],
    [11, 12, 14, 15, 16, 17, 18],
    [14, 15, 16, 17],
    [15, 16, 17],
    [10, 11, 12, 13, 14, 15, 18, 19, 20],
    [11, 12, 14, 18, 19, 20, 21],
    [18, 19, 20, 21],
    [19, 20, 21],
    [0, 1, 2, 4, 5, 8, 9],
]

EDGE_LIST = [
    (np.int64(0), 1),
    (np.int64(1), 2),
    (np.int64(2), 3),
    (np.int64(3), 4),
    (np.int64(0), 5),
    (np.int64(5), 6),
    (np.int64(6), 7),
    (np.int64(7), 8),
    (np.int64(0), 9),
    (np.int64(9), 10),
    (np.int64(10), 11),
    (np.int64(11), 12),
    (np.int64(12), 13),
    (np.int64(13), 14),
    (np.int64(12), 15),
    (np.int64(15), 16),
    (np.int64(16), 17),
    (np.int64(17), 18),
    (np.int64(12), 19),
    (np.int64(19), 20),
    (np.int64(20), 21),
    (np.int64(21), 22),
]

WINDOW_SIZE, BATCH_SIZE = 64, 256

def load_params(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        # validate_blocks(data)
    if data is None:
        return {}
    return data

def validate_blocks(config):
    """
    Validate a multi-block convolutional config.

    Required structure:
    config = {
        "block1": { "conv_params": {...}, ... },
        "block2": { "conv_params": {...}, ... },
        ...
    }
    """

    # Convert to ordered list of (block_name, block_data)
    blocks = list(config.items())

    # --------- Helper ----------
    def err(msg):
        raise ValueError(msg)

    # ===================================
    # Validate block-by-block
    # ===================================
    for idx, (block_name, block) in enumerate(blocks):
        if "conv_params" not in block:
            err(f"{block_name}: missing 'conv_params' section")

        p = block["conv_params"]

        # ---------- Extract parameters ----------
        try:
            in_c  = p["in_channels_per_joint"]
            out_c = p["out_channels_per_joint"]
            offset_in_c = p.get("offset_in_channels_per_joint", None)
            bias = p["bias"]
            kernel = p["kernel_size"]
            stride = p["stride"]
            padding = p["padding"]
            padding_mode = p["padding_mode"]
            dilation = p["dilation"]
            groups = p["groups"]
        except KeyError as e:
            err(f"{block_name}: missing conv parameter: {e}")

        # ===================================
        # Rule 1: Kernel size
        # ===================================
        if not isinstance(kernel, int) or kernel <= 0:
            err(f"{block_name}: kernel_size must be positive integer")
        if kernel % 2 == 0:
            err(f"{block_name}: kernel_size must be odd for your padding rule (got {kernel})")

        # ===================================
        # Rule 2: Padding matches kernel_size
        # ===================================
        expected_pad = (kernel - 1) // 2
        if padding != expected_pad:
            err(
                f"{block_name}: padding={padding} does not match (kernel_size - 1)//2 = {expected_pad}"
            )

        # ===================================
        # Rule 3: Stride
        # ===================================
        if stride < 1:
            err(f"{block_name}: stride must be >= 1 (got {stride})")

        # ===================================
        # Rule 4: Dilation
        # ===================================
        if dilation < 1:
            err(f"{block_name}: dilation must be >= 1 (got {dilation})")

        # ===================================
        # Rule 5: Groups divides channels
        # ===================================
        if in_c % groups != 0:
            err(
                f"{block_name}: in_channels_per_joint ({in_c}) must be divisible by groups ({groups})"
            )
        if out_c % groups != 0:
            err(
                f"{block_name}: out_channels_per_joint ({out_c}) must be divisible by groups ({groups})"
            )

        # ===================================
        # Rule 6: Padding mode validity
        # ===================================
        valid_padding_modes = {"constant", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            err(f"{block_name}: invalid padding_mode '{padding_mode}'")

        if padding_mode == "reflect" and padding <= 0:
            err(f"{block_name}: reflect padding_mode requires padding > 0")

        # ===================================
        # Rule 7: Bias is always allowed with groups=1
        # (PyTorch limitation: only grouped convs with mismatched channels can forbid bias)
        # No additional checks needed here.


    # ===================================
    # Validate cross-block linkage
    # ===================================
    for i in range(len(blocks) - 1):
        name_a, A = blocks[i]
        name_b, B = blocks[i+1]

        out_a = A["conv_params"]["out_channels_per_joint"]
        in_b  = B["conv_params"]["in_channels_per_joint"]

        if out_a != in_b:
            raise ValueError(
                f"Channel mismatch: {name_a}.out_channels_per_joint = {out_a} "
                f"but {name_b}.in_channels_per_joint = {in_b}"
            )


    # ===================================
    # Pooling structure rules
    # ===================================
    for i, (block_name, block) in enumerate(blocks):
        is_last = (i == len(blocks) - 1)

        pool      = block.get("pool", False)
        last_pool = block.get("last_pool", False)

        if last_pool and not is_last:
            err(f"{block_name}: last_pool=True is only allowed on the final block")

    # If all checks passed
    return True

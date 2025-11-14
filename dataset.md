# Dataset
## Motion Dataset
- Motion dataset length
    - `mlist_mdmval.pth`: A List containing 5823 data.
    - `mlist_mdmtrain.pth`: A List containing 5823 data.
    - `mlist_flame.pth`: A List containing 5823 data.
- Motion data format:
    - Each motion data is a Dict with two keys: `motion_better` and `motion_worse`
    - `motion_data["motion_better"]`: [frames, 25, 3], axis-angle with 24 SMPL joints and 1 XYZ root location

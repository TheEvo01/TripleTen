import numpy as np

spot_matrix = np.array(
    [
        [10, 11, 12, 13, 14, 15],
        [11, 12, 13, 14, 15, 16],
        [12, 13, 14, 15, 16, 17],
        [13, 14, 15, 16, 17, 18],
        [14, 15, 16, 17, 18, 19],
        [15, 16, 17, 18, 19, 20],
    ]
)

spot_counts = {}
total_spots = 6 * 6  # Total number of spots in the matrix

for i in range(6):
    for j in range(6):
        spot_value = spot_matrix[i][j]
        if spot_value not in spot_counts:
            spot_counts[spot_value] = 1
        else:
            spot_counts[spot_value] += 1

spot_probs = {k: v / total_spots for k, v in spot_counts.items()}

for i in range(10, 21):
    print(i, spot_probs.get(i, 0))  # Use get() to handle missing keys gracefully

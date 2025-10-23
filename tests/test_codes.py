import pandas as pd

nearest_neighbors = pd.DataFrame(
    {
        "latitude": [28.61, 28.52],
        "longitude": [77.23, 77.12],
        "2020-01-01": [0.45, 0.39],
        "2020-01-02": [0.48, None],
    }
)

print(nearest_neighbors)


# Testing the code to print each row's index and values

for ind, col in nearest_neighbors.iterrows():
    print(f"Row index: {ind}")
    print(f"Latitude: {col['latitude']}")
    print(f"longitude:{col['longitude']}")


# for idx, row in nearest_neighbors.iterrows():
#     print(f"Row index: {idx}")
#     print(f"Latitude: {row['latitude']}")
#     print(f"Longitude: {row['longitude']}")
#     print(f"AOD on 2020-01-01: {row['2020-01-01']}")
#     print(f"AOD on 2020-01-02: {row['2020-01-02']}")
#     print("-" * 30)

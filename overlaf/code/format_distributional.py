import bayopy

# Two actions with all 11 contexts: retain complete probability support.
columns = ['action_0', 'context_0', 'reference_probability', 'sampling_probability', 'margin', 'reward']
rows = [
    [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.1, 0.0, 0.0, 0.1, 0.010000000000000002],
    [0.0, 0.2, 0.0, 0.0, 0.1, 0.020000000000000004],
    [0.0, 0.3, 0.6875, 0.0, 0.1, 0.03],
    [0.0, 0.4, 0.1875, 0.0, 0.1, 0.04000000000000001],
    [0.0, 0.5, 0.08333333333333333, 1.0, 0.1, 0.05],
    [0.0, 0.6, 0.041666666666666664, 0.0, 0.1, 0.06],
    [0.0, 0.7, 0.0, 0.0, 0.1, 0.06999999999999999],
    [0.0, 0.8, 0.0, 0.0, 0.1, 0.08000000000000002],
    [0.0, 0.9, 0.0, 0.0, 0.1, 0.09000000000000001],
    [0.0, 1.0, 0.0, 0.0, 0.1, 0.1],
    [0.01, 0.0, 0.0, 0.0, 0.1, -0.05],
    [0.01, 0.1, 0.0, 0.0, 0.1, 0.019000000000000003],
    [0.01, 0.2, 0.0, 0.0, 0.1, 0.029000000000000005],
    [0.01, 0.3, 0.6875, 0.0, 0.1, 0.039],
    [0.01, 0.4, 0.1875, 0.0, 0.1, 0.04900000000000001],
    [0.01, 0.5, 0.08333333333333333, 1.0, 0.1, 0.059000000000000004],
    [0.01, 0.6, 0.041666666666666664, 0.0, 0.1, 0.06899999999999999],
    [0.01, 0.7, 0.0, 0.0, 0.1, 0.07899999999999999],
    [0.01, 0.8, 0.0, 0.0, 0.1, 0.08900000000000001],
    [0.01, 0.9, 0.0, 0.0, 0.1, 0.099],
    [0.01, 1.0, 0.0, 0.0, 0.1, 0.109],
]

directory = bayopy.new_run_directory("results/last_guide/format_distributional_wind")
path = bayopy.save_csv(bayopy.join_path(directory, "example.csv"), rows, columns=columns)
loaded = bayopy.load_dataset(path, environment="distributional")
print(loaded.preview(rows=4))
print("Action dimension:", loaded.bounds.dimension)
print("Rows:", len(loaded.dataset.observations))


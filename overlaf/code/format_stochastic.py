import bayopy

# Four actual rows from the external example table.
columns = [
    'action_0',
    'action_1',
    'action_2',
    'reward',
]
rows = [
    [
        0.5826446280991735,
        0.5492957746478874,
        0.5622169960335567,
        -151.0,
    ],
    [
        0.14876033057851243,
        0.352112676056338,
        0.22243672996595168,
        -75.0,
    ],
    [
        0.5165289256198347,
        0.43661971830985913,
        0.49657762645231474,
        -141.0,
    ],
    [
        0.30165289256198347,
        0.30985915492957744,
        0.5729228825160587,
        -206.0,
    ],
]

directory = bayopy.new_run_directory("results/last_guide/format_stochastic")
path = bayopy.save_csv(
    bayopy.join_path(directory, "example.csv"),
    rows, columns=columns,
)
loaded = bayopy.load_dataset(path, environment="stochastic")
print(loaded.preview(rows=4))
print("Action dimension:", loaded.bounds.dimension)
print("Rows:", len(loaded.dataset.observations))

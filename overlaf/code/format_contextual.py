import bayopy

# Four actual rows from the external example table.
columns = [
    'action_0',
    'action_1',
    'action_2',
    'context_0',
    'context_1',
    'context_2',
    'reward',
]
rows = [
    [
        0.8421052631578949,
        0.191699604743083,
        0.572192513368984,
        1.0,
        0.0,
        0.0,
        1065.0,
    ],
    [
        0.5710526315789473,
        0.2055335968379447,
        0.4171122994652407,
        1.0,
        0.0,
        0.0,
        1050.0,
    ],
    [
        0.5605263157894738,
        0.3201581027667984,
        0.7005347593582887,
        1.0,
        0.0,
        0.0,
        1185.0,
    ],
    [
        0.8789473684210525,
        0.2391304347826087,
        0.6096256684491979,
        1.0,
        0.0,
        0.0,
        1480.0,
    ],
]

directory = bayopy.new_run_directory("results/last_guide/format_contextual")
path = bayopy.save_csv(
    bayopy.join_path(directory, "example.csv"),
    rows, columns=columns,
)
loaded = bayopy.load_dataset(path, environment="contextual")
print(loaded.preview(rows=4))
print("Action dimension:", loaded.bounds.dimension)
print("Rows:", len(loaded.dataset.observations))

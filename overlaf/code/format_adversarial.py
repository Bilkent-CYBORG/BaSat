import bayopy

# Four actual rows from the external example table.
columns = [
    'action_0',
    'action_1',
    'action_2',
    'evaluated_action_0',
    'evaluated_action_1',
    'evaluated_action_2',
    'reward',
]
rows = [
    [
        0.0,
        0.423728813559322,
        0.375,
        0.0833333333333334,
        0.5084745762711864,
        0.375,
        6.0,
    ],
    [
        0.0833333333333334,
        0.5084745762711864,
        0.375,
        0.0,
        0.423728813559322,
        0.375,
        5.0,
    ],
    [
        0.0833333333333334,
        0.5932203389830508,
        0.5833333333333334,
        0.0833333333333334,
        0.6779661016949152,
        0.5833333333333334,
        6.0,
    ],
    [
        0.0833333333333334,
        0.6779661016949152,
        0.5833333333333334,
        0.0833333333333334,
        0.5932203389830508,
        0.5833333333333334,
        6.2,
    ],
]

directory = bayopy.new_run_directory("results/last_guide/format_adversarial")
path = bayopy.save_csv(
    bayopy.join_path(directory, "example.csv"),
    rows, columns=columns,
)
loaded = bayopy.load_dataset(path, environment="adversarial")
print(loaded.preview(rows=4))
print("Action dimension:", loaded.bounds.dimension)
print("Rows:", len(loaded.dataset.observations))

import bayopy

def stochastic_objective(values):
    target = (0.2, 0.65, 0.4)
    return -sum((x - t) ** 2 for x, t in zip(values, target))


def contextual_objective(values, context):
    c = float(context[0])
    target = (0.2 + 0.5 * c, 0.75 - 0.4 * c, 0.4)
    return -sum((x - t) ** 2 for x, t in zip(values, target))


def reveal_context(*, round_index, generator):
    return [float(round_index % 2)]


def distributional_objective(values, context):
    c = float(context[0])
    target = (0.25 + 0.35 * c, 0.6 - 0.2 * c, 0.3)
    distance = sum((x - t) ** 2 for x, t in zip(values, target))
    wave = 0.08 * bayopy.sin(6.0 * 3.141592653589793 * values[0])
    return float(-distance + wave)


def adversarial_objective(values):
    # This function receives the evaluated (already perturbed) coordinate.
    target = (0.3, 0.7, 0.5)
    return -sum((x - t) ** 2 for x, t in zip(values, target))


def apply_attack(action, *, round_index, generator):
    direction = 1.0 if round_index % 2 else -1.0
    changed = min(1.0, max(0.0, float(action[0]) + 0.05 * direction))
    return [changed - float(action[0]), 0.0, 0.0]

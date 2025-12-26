from flax.core.frozen_dict import FrozenDict, unfreeze

def process_variables(variables):
    """确保变量是 {'params': ...} 的格式，且没有多余的 'params' 套娃，且是普通 dict"""
    if isinstance(variables, FrozenDict):
        variables = unfreeze(variables)

    # 去除嵌套 params
    while isinstance(variables, dict) and "params" in variables:
        inner = variables["params"]
        if isinstance(inner, dict):
            variables = inner
        else:
            break

    return {"params": variables}

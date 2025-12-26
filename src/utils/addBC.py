from EAPINN import ICBC


def addbc(bc_config, geom):
    """
    根据 bc_config 和几何域 geom，批量初始化你的 IC/BC 对象列表。

    每个 bc_config 项应包含：
      - type:        'ic' | 'dirichlet' | 'neumann' | 'robin' | 'periodic' | 'operator' | 'pointset' | 'pointset_operator'
      - function:    真实值函数或算子函数
      - bc:          用于筛点的函数 (x_pt, flag) -> bool 或 mask
      - component:   （可选）指定输出通道，默认 0
      - component_x: （周期边界专用）周期方向索引
      - derivative_order: （周期边界专用）导数阶数，默认 0
      - points, values: （点集边界专用）坐标和真值
    """
    bcs = []
    for bc in bc_config:
        if bc.get('name') is None:
            bc['name'] = bc['type'] + ('' if bc['type'] == 'ic' else 'bc') + f"_{len(bcs) + 1}"
        if bc['type'] == 'dirichlet':
            bcs.append(ICBC.DirichletBC(geom, bc['function'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'robin':
            bcs.append(ICBC.RobinBC(geom, bc['function'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'ic':
            bcs.append(ICBC.IC(geom, bc['function'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'operator':
            bcs.append(ICBC.OperatorBC(geom, bc['function'], bc['bc']))
        elif bc['type'] == 'neumann':
            bcs.append(ICBC.NeumannBC(geom, bc['function'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'periodic':
            bcs.append(ICBC.PeriodicBC(geom, bc['component_x'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'pointset':
            bcs.append(ICBC.PointSetBC(bc['points'], bc['values'], component=bc['component']))
        else:
            raise ValueError(f"Unknown bc type: {bc['type']}")
    return bcs
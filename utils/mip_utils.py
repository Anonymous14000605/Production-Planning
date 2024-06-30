import pyscipopt as scip
import gurobipy as grb
from utils.ppgraph import PPGraph
import time
import os

"""
FORMULATION:
   node.mip_var:
       node.id: v nodes (total consumption), o nodes (binary: satisfied or not)
       node.id + parent.id: m-m edge, v-m edge
   vars:
       - o: binary, 1 if o satisfied, 0 otherwise
       - v:
           1. total requirement of v
           2. storage consumption of v

       - m:
           1.individual requirement of m
           2.production amount of m (set upper bound)
           3.storage consumption of m
   building cons (top->down):
       - m: sum(req_from(parent)) == req_to(child)
       - v: sum(req_from(parent)) - storage <= sum(req_to(child))
       - o: node.demand = req_to(child)
       
    obj_type:
        3 options: 'order_num', 'order_num_weighted', 'order_consumption_weighted'
        order_num:
            obj: p_(n-1) + p_(n-2) + ... + p_0
        order_num_weighted:
            obj: p_(n-1) + 2*p_(n-2) + ... + 3*p_k + ... + n*p_0
        order_consumption_weighted:
            obj: p_(n-1) + 2*p_(n-2) + ... + 3*p_k + ... + n*p_0 + m_(n-1) + 2*m_(n-2) + ... + 3*m_k + ... + n*m_0

"""


class MipSolvingConfig:
    def __init__(self, **kwargs):
        self.show_log=False, 
        self.time_limit=500,
        self.gap_limit=0, 
        self.use_presolve=True
        self.solver_type = 'scip'
        self.obj_type = 'order_num'
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'invalid attribute: {key}')

def formulate_mip_from_file(file_path, solving_config: MipSolvingConfig):
    if solving_config.solver_type == 'scip':
        model = scip.Model()
        model.readProblem(file_path)
        return model
    else:
        raise ValueError('solver type not supported')

def formulate_mip(solving_config: MipSolvingConfig, pp_graph: PPGraph):
    assert solving_config.solver_type in ['scip']
    _formulate_functions = {
        'scip': _formulate_mip_scip,
    }
    return _formulate_functions[solving_config.solver_type](obj_type=solving_config.obj_type, pp_graph=pp_graph)


def solve_mip(model, solving_config: MipSolvingConfig, pp_graph: PPGraph = None):
    assert solving_config.solver_type in ['scip']
    _solve_functions = {
        'scip': _solve_mip_scip,
    }
    return _solve_functions[solving_config.solver_type](model, solving_config, pp_graph)

def restore_real_obj(mip_results, solving_config: MipSolvingConfig, pp_graph: PPGraph):
    # restore real obj: product num
    real_obj = 0
    for product_node in pp_graph.root.children:
        real_obj += product_node.value * mip_results['solution'][product_node.id]
    # product num
    for node in pp_graph.nodes:
        if node.node_type in ['material', 'group']:
            for p in node.parents:
                real_obj -= 0.001 * mip_results['solution'][node.id + '_to_' + p.id]
        
    return real_obj


def output_mip_file(model: grb.Model, save_dir, file_name):
    model.setParam('OutputFlag', 0)   # hide output
    saved_path = os.path.join(save_dir, file_name + '.mps')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.write(saved_path)


def _formulate_mip_scip(obj_type, pp_graph):
    model = scip.Model()

    ## Define variables for MIP
    for node in pp_graph.nodes:
        var_dict_for_node = {}
        if node.node_type == 'root':
            pass
        elif node.node_type == 'group':
            for p_node in node.parents:
                var_dict_for_node[node.id + '_to_' + p_node.id] = model.addVar(node.id, vtype='INTEGER')
            # storage consumption
            var_dict_for_node[node.id + '_storage'] = model.addVar(node.id, vtype='INTEGER')
        elif node.node_type == 'material':
            # requirement from parents
            for p_node in node.parents:
                var_dict_for_node[node.id + '_to_' + p_node.id] = model.addVar(node.id + '_to_' + p_node.id, vtype='INTEGER')
            # production amount
            var_dict_for_node[node.id + '_production'] = model.addVar(node.id, vtype='INTEGER')
            # storage consumption
            var_dict_for_node[node.id + '_storage'] = model.addVar(node.id, vtype='INTEGER')

        elif node.node_type == 'order':
            var_dict_for_node[node.id] = model.addVar(node.id, vtype='BINARY')
        else:
            raise ValueError('node type not supported')
        node.mip_var_dict = var_dict_for_node

    ## set objective
    # order num + priority + material consumption
    if obj_type == 'order_consumption_weighted':
        p_mult_ratio = 10000
        model.setObjective(
            scip.quicksum(order_node.mip_var_dict[order_node.id] * order_node.order_priority for order_node in pp_graph.root.children) * p_mult_ratio +
            scip.quicksum(- node.priority * node.mip_var_dict[node.id + '_storage'] for node in pp_graph.nodes if node.node_type == 'material'),
            sense='maximize'
        )
    # product num
    elif obj_type == 'product_num':
        model.setObjective(scip.quicksum(product_node.mip_var_dict[product_node.id] * product_node.value * product_node.weight for product_node in pp_graph.root.children), sense='maximize')
    # product num - 0.001 * sum(every material's utilization amount for other produces)
    elif obj_type == 'product_minus_production':
        model.setObjective(
            scip.quicksum(product_node.mip_var_dict[product_node.id] * product_node.value * product_node.weight for product_node in pp_graph.root.children) 
            - 0.001 * scip.quicksum(node.mip_var_dict[node.id + '_to_' + p_node.id] if node.node_type in ['material', 'group'] else 0 for node in pp_graph.nodes for p_node in node.parents )
            , sense='maximize'
        )
    else:
        raise NotImplementedError

    ## add cons
    # 3 types of cons: vm-sub_tree; v-m; m
    for node in pp_graph.nodes:
        if node.node_type == 'root':
            continue

        elif node.node_type == 'group':
            # production amount: leq for 'product_minus_production'
            if obj_type == 'product_minus_production':
                model.addCons(
                    scip.quicksum(node.mip_var_dict[node.id + '_to_' + p_node.id] for p_node in node.parents)
                    <=
                    scip.quicksum(c_node.mip_var_dict[c_node.id + '_to_' + node.id] for c_node in node.children)
                    + node.mip_var_dict[node.id + '_storage']
                )
            else:
                model.addCons(
                    scip.quicksum(node.mip_var_dict[node.id + '_to_' + p_node.id] for p_node in node.parents)
                    ==
                    scip.quicksum(c_node.mip_var_dict[c_node.id + '_to_' + node.id] for c_node in node.children)
                    + node.mip_var_dict[node.id + '_storage']
                )

            # storage
            # model.addCons(node.mip_var_dict[node.id + '_storage'] <= node.storage)

        elif node.node_type == 'order':
            # essential requirements
            model.addCons(node.children[0].mip_var_dict[node.children[0].id + '_to_' + node.id]
                          == node.value * node.mip_var_dict[node.id])

        elif node.node_type == 'material':
            # basic equations: sum(req_from(parent)) <= storage + production amount
            if obj_type == 'product_minus_production':
                model.addCons(
                    scip.quicksum(node.mip_var_dict[node.id + '_to_' + parent.id] * parent.get_edge_weight(node.id) for parent in node.parents)
                    <= node.mip_var_dict[node.id + '_production'] + node.mip_var_dict[node.id + '_storage']
                )
            else:
                model.addCons(
                    scip.quicksum(node.mip_var_dict[node.id + '_to_' + parent.id] * parent.get_edge_weight(node.id) for parent in node.parents)
                    == node.mip_var_dict[node.id + '_production'] + node.mip_var_dict[node.id + '_storage'] 
                )

            # essential requirements: req_to(child) = production amount
            for c_node in node.children:
                model.addCons(c_node.mip_var_dict[c_node.id + '_to_' + node.id] == node.mip_var_dict[node.id + '_production'])

            # produce upper bound
            model.addCons(node.mip_var_dict[node.id + '_production'] <= node.produce_ub)

            # storage upper bound
            model.addCons(node.mip_var_dict[node.id + '_storage'] <= node.value)

        else:
            raise NotImplementedError

    return model


def _formulate_mip_gurobi_deprecated(obj_type, pp_graph):
    model = grb.Model()
    model.setParam('OutputFlag', 0)

    # add variables
    # add var
    # Notice: one node is visited once, as it appears once in the list
    for node in pp_graph.nodes:
        if node.node_type == 'root':
            pass

        elif node.node_type == 'group':
            node.mip_var_dict[node.id] = model.addVar(name=node.id, vtype=grb.GRB.INTEGER)
            # storage consumption
            node.mip_var_dict[node.id + '_storage'] = model.addVar(name=node.id, vtype=grb.GRB.INTEGER)

        elif node.node_type == 'material':
            # requirement from parents
            for p_node in node.parents:
                node.mip_var_dict[node.id + p_node.id] = \
                    model.addVar(name=node.id + '_to_' + p_node.id, vtype=grb.GRB.INTEGER)
            # production amount
            node.mip_var_dict[node.id + '_production'] = \
                model.addVar(name=node.id, vtype=grb.GRB.INTEGER)
            # storage consumption
            node.mip_var_dict[node.id + '_storage'] = \
                model.addVar(name=node.id, vtype=grb.GRB.INTEGER)

        elif node.node_type == 'order':
            node.mip_var_dict[node.id] = model.addVar(name=node.id, vtype=grb.GRB.BINARY)

        else:
            raise ValueError('node type not supported')
    model.update()

    # add constraints
    for node in pp_graph.nodes:
        if node.node_type == 'root':
            continue

        elif node.node_type == 'group':
            # production amount
            model.addConstr(
                grb.quicksum(c_node.mip_var_dict[c_node.id + '_to_' + node.id] if c_node.node_type == 'material' else
                             c_node.mip_var_dict[c_node.id] for c_node in node.children)
                + node.mip_var_dict[node.id + '_storage']
                == node.mip_var_dict[node.id]
            )

            # storage
            # model.addConstr(node.mip_var_dict[node.id + '_storage'] <= node.storage)

        elif node.node_type == 'order':
            # essential requirements
            model.addConstr(node.children[0].mip_var_dict[node.children[0].id + '_to_' + node.id]
                            == node.value * node.mip_var_dict[node.id])

        elif node.node_type == 'material':
            # basic equations: sum(req_from(parent)) <= storage + production amount
            model.addConstr(grb.quicksum(node.mip_var_dict[node.id + '_to_' + parent.id] for parent in node.parents)
                            * node.consuming_rate
                            ==
                            node.mip_var_dict[node.id + '_production'] + node.mip_var_dict[node.id + '_storage'])

            # essential requirements: req_to(child) = production amount
            for c_node in node.children:
                if c_node.node_type == 'material':
                    model.addConstr(c_node.mip_var_dict[c_node.id + '_to_' + node.id] ==
                                    node.mip_var_dict[node.id + '_production'])
                else:
                    model.addConstr(c_node.mip_var_dict[c_node.id] == node.mip_var_dict[node.id + '_production'])

            # produce upper bound
            model.addConstr(node.mip_var_dict[node.id + '_production'] <= node.produce_ub)

            # storage upper bound
            model.addConstr(node.mip_var_dict[node.id + '_storage'] <= node.value)

        else:
            raise NotImplementedError

    # set objective
    if obj_type != 'product_num':
        raise NotImplementedError
    # product num
    model.setObjective(grb.quicksum(product_node.mip_var_dict[product_node.id] * product_node.value
                                    for product_node in pp_graph.root.children), sense=grb.GRB.MAXIMIZE)

    return model


def _solve_mip_scip(model, solving_config: MipSolvingConfig, pp_graph: PPGraph = None):
    # settings
    if solving_config:
        if not solving_config.show_log:
            model.hideOutput()
        model.setRealParam('limits/time', solving_config.time_limit)
        if not solving_config.use_presolve:
            model.setParam('lp/presolving', False)
            model.setParam('presolving/maxrounds', 0)
            model.setParam('presolving/maxrestarts', 0)
        # model.setRealParam('limits/gap', solving_config.gap_limit)

    # record time
    start_time = time.time()
    # solve
    model.optimize()
    solving_time = time.time() - start_time

    ## record results
    mip_results = {
        'obj_value': model.getObjVal(),
        'solving_time': solving_time,
        'solution': {}
    }
    if pp_graph:
        # solution
        mip_results['solution'] = {}
        for node in pp_graph.nodes:
            if node.node_type == 'root':
                continue
            elif node.node_type == 'group':
                # storage consumption
                mip_results['solution'][node.id + '_storage'] = model.getVal(node.mip_var_dict[node.id + '_storage'])
                # total consumption
                # mip_results['solution'][node.id] = model.getVal(node.mip_var_dict[node.id])
                for p_node in node.parents:
                    mip_results['solution'][node.id + '_to_' + p_node.id] = model.getVal(node.mip_var_dict[node.id + '_to_' + p_node.id])
            elif node.node_type == 'material':
                # production amount
                mip_results['solution'][node.id + '_production'] = model.getVal(node.mip_var_dict[node.id + '_production'])
                # storage consumption
                mip_results['solution'][node.id + '_storage'] = model.getVal(node.mip_var_dict[node.id + '_storage'])
                # individual requirement
                for p_node in node.parents:
                    mip_results['solution'][node.id + '_to_' + p_node.id] = model.getVal(node.mip_var_dict[node.id + '_to_' + p_node.id])
            elif node.node_type == 'order':
                mip_results['solution'][node.id] = model.getVal(node.mip_var_dict[node.id])
            else:
                raise ValueError('node type not supported')

    return mip_results


def _solve_mip_gurobi_deprecated(model, solving_config: MipSolvingConfig, pp_graph: PPGraph = None):
    # settings
    if solving_config:
        if not solving_config.show_log:
            model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', solving_config.time_limit)
        if not solving_config.use_presolve:
            model.setParam('Presolve', 0)
        # gap
        model.setParam('MIPGap', solving_config.gap_limit)

    # record solving time
    start_time = time.time()
    model.optimize()
    solving_time = time.time() - start_time

    # output
    results = {
        'solving_time': solving_time,
        'obj': model.objVal
    }

    return results


def _formulate_mip_scip_online(obj_type, pp_graph):
    model = scip.Model()

    # add var
    # Notice: one node is visited once, as it appears once in the list
    for node in pp_graph.nodes:
        if node.node_type == 'root':
            pass

        elif node.node_type == 'group':
            node.mip_var_dict[node.id] = model.addVar(node.id, vtype='INTEGER')
            # storage consumption
            node.mip_var_dict[node.id + '_storage'] = model.addVar(node.id, vtype='INTEGER')

        elif node.node_type == 'material':
            # requirement from parents
            for p_node in node.parents:
                node.mip_var_dict[node.id + p_node.id] = model.addVar(node.id + p_node.id, vtype='INTEGER')
            # production amount
            node.mip_var_dict[node.id + '_production'] = model.addVar(node.id, vtype='INTEGER')
            # storage consumption
            node.mip_var_dict[node.id + '_storage'] = model.addVar(node.id, vtype='INTEGER')

        elif node.node_type == 'order':
            node.mip_var_dict[node.id] = model.addVar(node.id, vtype='BINARY')

        else:
            raise ValueError('node type not supported')

    # set obj
    # order num
    if obj_type == 'order_num':
        model.setObjective(scip.quicksum(order_node.mip_var_dict[order_node.id]
                                         for order_node in pp_graph.root.children), sense='maximize')
    # order num + priority
    elif obj_type == 'order_num_weighted':
        model.setObjective(scip.quicksum(order_node.mip_var_dict[order_node.id] * order_node.order_priority
                                         for order_node in pp_graph.root.children), sense='maximize')
    # order num + priority + material consumption
    elif obj_type == 'order_consumption_weighted':
        p_mult_ratio = 10000
        model.setObjective(scip.quicksum(order_node.mip_var_dict[order_node.id] * order_node.order_priority
                                         for order_node in pp_graph.root.children) * p_mult_ratio +
                           scip.quicksum(- node.priority * node.mip_var_dict[node.id + '_storage']
                                         for node in pp_graph.nodes if node.node_type == 'material'),
                           sense='maximize')
    # product num
    elif obj_type == 'product_num':
        model.setObjective(scip.quicksum(product_node.mip_var_dict[product_node.id] * product_node.value
                                         for product_node in pp_graph.root.children), sense='maximize')
    elif obj_type == 'priority':
        model.setObjective(scip.quicksum(order_node.mip_var_dict[order_node.id]
                                         for order_node in pp_graph.root.children), sense='maximize')
        order_nodes = pp_graph.root.children
        # sort by priority
        order_nodes.sort(key=lambda x: x.order_priority, reverse=True)
        # add cons
        for i in range(len(order_nodes) - 1):
            model.addCons(order_nodes[i].mip_var_dict[order_nodes[i].id] >= order_nodes[i + 1].mip_var_dict[
                order_nodes[i + 1].id])
    else:
        raise NotImplementedError

    # add cons
    # 3 types of cons: vm-sub_tree; v-m; m
    for node in pp_graph.nodes:
        if node.node_type == 'root':
            continue

        elif node.node_type == 'group':
            # production amount
            model.addCons(
                scip.quicksum(c_node.mip_var_dict[c_node.id + node.id] if c_node.node_type == 'material' else
                              c_node.mip_var_dict[c_node.id] for c_node in node.children)
                + node.mip_var_dict[node.id + '_storage']
                == node.mip_var_dict[node.id]
            )

            # storage
            model.addCons(node.mip_var_dict[node.id + '_storage'] <= node.value)

        elif node.node_type == 'order':
            # essential requirements
            model.addCons(node.children[0].mip_var_dict[node.children[0].id + node.id]
                          == node.value * node.mip_var_dict[node.id])

        elif node.node_type == 'material':
            # basic equations: sum(req_from(parent)) <= storage + production amount
            model.addCons(scip.quicksum(node.mip_var_dict[node.id + parent.id] for parent in node.parents)
                          * node.consuming_rate
                          ==
                          node.mip_var_dict[node.id + '_production'] + node.mip_var_dict[node.id + '_storage'])

            # essential requirements: req_to(child) = production amount
            for c_node in node.children:
                if c_node.node_type == 'material':
                    model.addCons(c_node.mip_var_dict[c_node.id + node.id] ==
                                  node.mip_var_dict[node.id + '_production'])
                else:
                    model.addCons(c_node.mip_var_dict[c_node.id] == node.mip_var_dict[node.id + '_production'])

            # produce upper bound
            model.addCons(node.mip_var_dict[node.id + '_production'] <= node.produce_ub)

            # storage upper bound
            model.addCons(node.mip_var_dict[node.id + '_storage'] <= node.value)

        else:
            raise NotImplementedError

    return model
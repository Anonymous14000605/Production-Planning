from dataclasses import dataclass
import pickle


@ dataclass
class Node:
    def __init__(self):
        self._parents = []
        self._children = []

    def _add_parent(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        self._parents.extend(nodes)
    
    def _add_child(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        self._children.extend(nodes)
    
    @ classmethod
    def add_edge(cls, parents, children):
        if not isinstance(parents, list):
            parents = [parents]
        if not isinstance(children, list):
            children = [children]
        for parent in parents:
            parent._add_child(children)
        for child in children:
            child._add_parent(parents)
    
    @ property
    def parents(self):
        return self._parents
    
    @ property
    def children(self):
        return self._children
    
    def remove_parents(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        for node in nodes:
            self._parents.remove(node)
    
    def remove_children(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        for node in nodes:
            self._children.remove(node)
    
    def remove_all_children(self):
        self._children = []
    
    def remove_all_parents(self):
        self._parents = []


@ dataclass
class PPNode(Node):
    def __init__(self, node_type, id, **kwargs):
        super().__init__()
        self._node_type = node_type
        self._id = id
        assert self._node_type in ['root', 'material', 'order', 'group']
        self._allocate_attributes(**kwargs)

    def _allocate_attributes(self, **kwargs):
        # distinctive attributes
        if self._node_type == 'material':
            self._storage = kwargs['storage']
        elif self._node_type == 'order':
            self._demand = kwargs['demand']
        elif self._node_type == 'group':
            pass

        # common attributes
        self._produce_attrs = {
            'lb': 0,
            'ub': 0,
            'consuming_rate': 1, # not used
            'weight': 1
        }
        
        # mip attributes
        self._mip_attrs = {
            'var': None
        }

        # visited flag
        self._visited = False

        # attributes for problem construction
        if 'depth' in kwargs:
            self._depth = kwargs['depth']
        else:
            self._depth = None

    def __eq__(self, other):
        if isinstance(other, PPNode):
            return self._id == other.id
        return False
    
    def __repr__(self):
        return 'node_{}'.format(self._id)
    
    def __hash__(self):
        return hash(self._id)

    def reset(self):
        self._visited = False
        self._mip_attrs = {
            'var': None
        }

    @ property
    def id(self):
        return self._id

    @ property
    def visited(self):
        return self._visited

    @ property
    def node_type(self):
        return self._node_type
    
    @ property
    def value(self):
        # storage for material, demand for order, none for group
        if self._node_type == 'material':
            return self._storage
        elif self._node_type == 'order':
            return self._demand
        elif self._node_type == 'group':
            return None
        
    @ property
    def weight(self):
        return self._produce_attrs['weight']
    
    def get_edge_weight(self, node_id):
        return 1.0
        # for idx, c in enumerate(self.children):
        #     if c.id == node_id:
        #         return self._edge_weights[idx]
        # raise ValueError('node not found')

    @ property
    def consuming_rate(self):
        return self._produce_attrs['consuming_rate']
    
    @ property
    def produce_ub(self):
        return self._produce_attrs['ub']

    @ property
    def depth(self):
        return self._depth
    
    @ property
    def mip_var_dict(self):
        return self._mip_attrs['var']
    
    @ mip_var_dict.setter
    def mip_var_dict(self, value):
        self._mip_attrs['var'] = value

    @ visited.setter
    def visited(self, value):
        self._visited = value

    @ weight.setter
    def weight(self, value):
        self._produce_attrs['weight'] = value


@ dataclass
class PPGraph:
    def __init__(self, root=None, file=None):
        # either root or file should be provided; but not both
        if root is None and file is None:
            raise ValueError('either root or file should be provided')
        elif root is not None and file is not None:
            raise ValueError('only one of root and file should be provided')
        
        if root:
            self._root = root
            assert type(root) == PPNode
        else:
            with open(file, 'rb') as f:
                graph = pickle.load(f)
            self._root = graph._root
        # get node list
        self._nodes = set()
        self._get_nodes(self._root, file)
        self.add_edge_weight_attrs()
    
    # update node list; avoid repeated entries
    def _get_nodes(self, node, file):
        self._nodes.add(node)
        # if file is provided: connect children to their parents
        if file:
            for child in node.children:
                child._add_parent(node) 
        # add to set
        for child in node.children:
            if child not in self._nodes:
                self._get_nodes(child, file)

    # append weight attrs: list[int(default 1)] to all nodes; add '_edge_weights' attr to nodes
    def add_edge_weight_attrs(self):
        for node in self._nodes:
            node._edge_weights = [1] * len(node.children)
    
    @ property
    def nodes(self):
        # convert to list
        return list(self._nodes)
    @ property
    def orders(self):
        return self._root.children
    
    @ property
    def root(self):
        return self._root
    
    def reset(self):
        for node in self._nodes:
            node.reset()
    
    def update_produce_ub(self):
        # recursively calc ub for every node
        def calc_ub_for_node(node):
            # no children
            if not node.children:
                node._produce_attrs['ub'] = 0
                return

            # calc for children
            for c in node.children:
                if c.visited:
                    continue
                else:
                    c._visited = True
                    calc_ub_for_node(c)
            # calc for node
            if node.node_type == 'material':
                node._produce_attrs['ub'] = min([c._produce_attrs['ub'] + c.value if c.node_type == 'material' else c._produce_attrs['ub'] for c in node.children])
            elif node.node_type == 'group':
                node._produce_attrs['ub'] = sum([c._produce_attrs['ub'] + c.value if c.node_type == 'material' else c._produce_attrs['ub'] for c in node.children])
            else:
                # root, order
                node._produce_attrs['ub'] = 0

        # calc for every sub tree
        for order in self._root.children:
            product_node = order.children[0]
            if not product_node.visited:
                product_node.visited = True
                calc_ub_for_node(product_node)

        # reset
        self.reset()

    # delete the current graph completely
    def delete(self):
        if self._root not in self._nodes:
            self._nodes.append(self._root)
        for node in self._nodes:
            node.remove_parents(node.parents)
            node.remove_children(node.children)
            # delete all attributes, set to None
            for attr in node.__dict__:
                node.__dict__[attr] = None
        self._nodes = []
        self._root = None

    # output as pkl file
    def save(self, path):
        # delete node.parents for all nodes
        self.reset()
        for node in self._nodes:
            node.remove_all_parents()
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    # show the graph
    def display(self):
        def show_node(node):
            # display: id, value, depth, parents, children
            print('id: {}, value: {}, ub:{}, depth: {}, parents: {}, children: {}'.format(node.id, node.value, node._produce_attrs['ub'], node.depth, [p.id for p in node.parents], [c.id for c in node.children]))

        for node in self._nodes:
            show_node(node)

    # adjust weight
    def adjust_weights(self, weights, weight_type='order'):
        if type(weights) == tuple:
            order_weights, material_weights = weights
            order_weights = order_weights.cpu().numpy()
            material_weights = material_weights.cpu().numpy()
        else:
            if type(weights) == torch.Tensor:
                weights = weights.cpu().numpy()
            if 'order' in weight_type:
                order_weights = weights
            elif 'material' in weight_type:
                material_weights = weights
            else:
                raise ValueError('invalid weight type')
        
            
        # sort by id
        if 'order' in weight_type:
            sorted_nodes = sorted([node for node in self.nodes if node.node_type == 'order'], key=lambda x: x.id)
            assert len(sorted_nodes) == len(weights)
            # assign weight
            for node, weight in zip(sorted_nodes, order_weights):
                node.weight = weight
        if 'material' in weight_type:
            # update edge weights for material-material edges
            weight_idx = 0
            for node in self._nodes:
                if node.node_type == 'material':
                    for c_idx, c in enumerate(node.children):
                        if c.node_type == 'material':
                            node._edge_weights[c_idx] = material_weights[weight_idx]
                            weight_idx += 1
            assert weight_idx == len(material_weights)
        if not 'order' in weight_type and not 'material' in weight_type:
            raise ValueError('not implemented')



        

    
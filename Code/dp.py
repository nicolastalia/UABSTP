import networkx as nx
import matplotlib.pyplot as plt
import copy
import dp
import scenarios_generator as sg
import states as st

# From ChatGPT
class TreeNode:
    def __init__(self, value,stage,state,node_type,node_index):
        self.value = value
        self.children = []  # list of tuples (child_node, edge_value)
        self.stage = stage
        self.state = state
        self.node_type = node_type
        self.to_go = None
        self.node_index = node_index

    def __str__(self):
        return f'index: {self.node_index} stage: {self.stage},type: {self.node_type}, value: {self.value}, state: {self.state}, go: {self.to_go} '

    def add_child(self, child_node, edge_value):
        self.children.append((child_node, edge_value))
    
    def get_min_child_value(self):
        if not self.children:
            return None  # or float('inf') if you want to use it in min comparisons

        # Get the minimum value among the children
        min_value = float('inf')
        for child, edge_value in self.children:
            if (edge_value + child.value) < min_value:
                min_value = edge_value + child.value
                go_to = child.node_index
        return min_value, go_to
def add_edges(graph, node, parent=None):
    for child, edge_value in node.children:
        graph.add_edge(id(node), id(child), weight=edge_value)
        add_edges(graph, child, node)


class DpTree:
    def __init__(self,start=None,working_hours=None,decision_interval=None,node_index = 1,centres=None,vehicles=None,decision=None,scenario=None):
        sample_path = {}
        sample_path[1,'centre_1']= scenario

        state = st.state('st1',vehicles,centres)
        list = []
        list.append(TreeNode(0,480,state,'pre',node_index))

        for i in range(start,start+working_hours,10):
            temporary_pre = [nod for nod in list if nod.stage == i and nod.node_type == 'pre']
            print(f'iteration: {i}, number of nodes: {len(temporary_pre)}')
            for node in temporary_pre:
                # print(node)
                if node.state.vehicles['vehicle_1'].vlab == 0: # if vehicle is in the lab
                    # dispatch the vehicle only if there is any sample in the centre
                    if node.state.centres['centre_1'].req > 0:
                        temp_node = copy.deepcopy(node)
                        node_index += 1
                        temp_node.node_index = node_index
                        temp_node.state.update('vehicle_1',decision)
                        # temp_node.value = 0
                        temp_node.children = []
                        temp_node.node_type = 'pos'
                        list.append(temp_node)
                        node.add_child(temp_node,decision['lab'])
                    # keep the vehicle in the lab
                    if node.state.centres['centre_1'].crit >= 30:
                        temp_node = copy.deepcopy(node)
                        node_index += 1
                        temp_node.node_index = node_index
                        # temp_node.value = 0
                        temp_node.children = []
                        temp_node.node_type = 'pos'
                        list.append(temp_node)
                        node.add_child(temp_node,0)
                    else:
                        temp_node = copy.deepcopy(node)
                        temp_node.state.centres['centre_1'].req = 0
                        temp_node.state.centres['centre_1'].crit = 1440
                        node_index += 1
                        temp_node.node_index = node_index
                        # temp_node.value = 0
                        temp_node.children = []
                        temp_node.node_type = 'pos'
                        list.append(temp_node)
                        node.add_child(temp_node,10440)
                if node.state.vehicles['vehicle_1'].vlab > 0: # if vehicle is on a trip
                    # keep the vehicle following its route
                    temp_node = copy.deepcopy(node)
                    node_index += 1
                    temp_node.node_index = node_index
                    if node.state.centres['centre_1'].crit >= 30:
                        temp_node.value = 0
                    else:
                        temp_node.value = 100
                    temp_node.children = []
                    temp_node.node_type = 'pos'
                    list.append(temp_node)
                    node.add_child(temp_node,0)
            temporary_pos = [nod for nod in list if nod.stage == i and nod.node_type == 'pos']
            for node in temporary_pos:
                temp_node = copy.deepcopy(node)
                node_index += 1
                temp_node.node_index = node_index
                temp_node.state.state_update_pre(decision_interval,i,i+decision_interval,sample_path,1)
                # temp_node.value = 0
                temp_node.children = []
                temp_node.node_type = 'pre'
                temp_node.stage += decision_interval
                list.append(temp_node)
                node.add_child(temp_node,0)

        self.list = list
        self.last_stage = start + working_hours
        self.decision_interval = decision_interval
        self.start = start

    def optimise(self):
        last_stage = self.last_stage - self.decision_interval
        print(last_stage)
        interval = self.decision_interval
        current_stage = last_stage
        nodes_types = ['pos','pre']

        while current_stage >= self.start:
            # print(current_stage)
            for typ in nodes_types:
                updating_list = [i for i in self.list if i.stage == current_stage and i.node_type == typ]

                for i in updating_list:
                    update_value, go = i.get_min_child_value()
                    # print(i.stage,i.state,i.node_type, i.children,i.value,i.get_min_child_value())
                    i.value = update_value
                    i.to_go = go
                    # print(i.value)
            current_stage += - interval
        return self


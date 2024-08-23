# importing libraries
from geopy.geocoders import Nominatim
import ssl
import certifi
import folium
import requests
import numpy as np
import json
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from collections import Counter
import copy
import networkx as nx
import random
import math
import operator

import neuradp as nadp



os.environ['GRB_LICENSE_FILE'] = 'c9697abb-f79e-46a9-b050-2895ccc98a02'
env = gp.Env(empty=True)
env.start()


class transition:
    def __init__(self, statepre, statepos,rout,realization):
        self.statepre = statepre
        self.statepos = statepos
        self.rout = rout
        self.realization = realization
        # self.reward = reward

class Experience:
    def __init__(self,pre_state=None,pos_state=None,reward=None,value_function=None,time=None,action=None,realization=None,historic_action=None):
        self.state = state
        self.time = time
        self.action = action
        self.realization = realization
        self.pre_state = pre_state
        self.pre_state_str = str(pre_state)
        self.pos_state = pos_state
        self.reward = reward
        self.value_function = value_function
        self.visited = 1
        self.historic_actions = historic_action


class Centre:
    def __init__(self,name,req = 0,crit=1440):
        self.req = req
        self.crit = crit
        self.name = 'centre_'+ str(name)
    def __str__(self):
        return f'({self.name}: [{self.req}, {self.crit}])'

class Vehicle:
    def __init__(self,name,vlab,centres,type='Private'):
        dict = {}
        for centre_key,centre_value in centres.items():
            dict[centre_key] = 1440
        self.vname = 'vehicle_' + name
        self.vlab = vlab
        self.vcent = dict if type=='Private' else None
        self.value_acum = 0
        self.value_function = 0
        self.type = type
        self.times_dispatched = 0
        self.external_dispatched = False

    def __str__(self):
        return f'({self.vname}: [lab: {self.vlab}, {self.vcent}])'


class state: 
    nnmodel = None
    value = 0
    expired = 0
    
    def restart_nn():
        state.nnmodel = None

    def restart_global_attributes():
        state.value = 0
        state.expired = 0      

    def __init__(self,name,vehicles={},centres={},duration={},epoch_interval = 10):
        self.name = name
        self.vehicles = vehicles
        self.centres = centres
        self.time = 0
        self.duration = duration
        state.nnmodel = nadp.NeurADP(len(self.centres))
        self.total_duration = 0
        self.value_function = {}
        self.action = []
        self.realization = 0
        self.immediate_cost = 0
        self.actual_value = 0
        self.epoch_interval = epoch_interval
        self.external_used = 0
        self.finishing_workshift = 960
        self.max_travel_duration = 120
        self.historic_actions = []

    def restart_state(self):
        for centre_name, centre_obj in self.centres.items():
            centre_obj.req = 0
            centre_obj.crit = 1440
        for vehicle_name, vehicle_obj in self.vehicles.items():
            vehicle_obj.vlab = 0
            vehicle_obj.value_acum = 0
            if vehicle_obj.type == 'Private':
                for cent in self.centres:
                    vehicle_obj.vcent[cent] = 1440
            else: 
                vehicle_obj.vcent = None
        self.total_duration = 0
        self.external_used = 0
        self.value_function = {}
        self.historic_actions = []
        return self
    
    def __str__(self):
        veh_out = 'vehicles: '
        cent_out = 'centres: '
        for i,j in self.vehicles.items():
            veh_out += f'{j}, '
        for i,j in self.centres.items():
            cent_out += f'{j}, '
        return f'({self.name}_{self.time}: {veh_out}{cent_out})'
    
    def update_priv(self,vehicle,action,trainning=False,individual=False):
        self.vehicles[vehicle].vlab = action['lab']
        self.total_duration += 0 if individual else action['lab']
        if trainning == False:
            self.vehicles[vehicle].value_acum += action['lab']
        for centre_index, centre_value in action.items():
            if centre_index != 'lab':
                self.vehicles[vehicle].vcent[centre_index] = action[centre_index]
                self.centres[centre_index].req = 0
                self.centres[centre_index].crit = 1440
        return self
    
    def update_ext(self,vehicle,individual=False):
        self.vehicles[vehicle].vlab = 0
        centre = vehicle.split('_', 1)[1]
        self.total_duration += 0 if individual else ((self.duration['lab',centre] + self.duration[centre,'lab'])*2)
        self.centres[centre].req = 0
        self.centres[centre].crit = 1440
        self.vehicles[vehicle].times_dispatched += 1
        self.vehicles[vehicle].external_dispatched = True
        self.external_used += 1
        return self

    def centres_req(self):
        
        centwreq = []
        centwreq_crit = []
        for i,j in self.centres.items():
            if j.req > 0:
                dur = self.duration['lab',j.name] + self.duration[j.name,'lab']  
                if dur >= j.crit - self.epoch_interval:
                    centwreq_crit.append(j.name)
                else:
                    centwreq.append(j.name)
        return centwreq, centwreq_crit
    
    def vehicles_list(self):
        vehicles_array = []
        for i,j in self.vehicles.items():
            if j.vlab == 0:
                vehicles_array.append(j.vname)
        return vehicles_array

    def routes(self):
        centwreq = [j.name for i,j in self.centres.items() if j.req > 0]

        b = len(centwreq)
        list_perm = []
        dict_perm = {}
        count = 0
        # Generate all permutations
        while b >= 1:
            permutations = list(itertools.permutations(centwreq,b))

            # Print the permutations
            for perm in permutations:
                route = [i for i in perm]
                route.insert(0,'lab')
                route.append('lab')
                rut = {}
                current = 0
                for r in range(len(route)-1):
                    current += self.duration[route[r],route[r+1]]
                    rut[route[r+1]] = current
                for i in route[1:-1]:
                    if (rut['lab'] <= self.centres[i].crit) and ((rut['lab'] - rut[self.centres[i].name])<=self.max_travel_duration):
                        count += 1
                        list_perm.append(rut)
                        dict_perm[f'route_{count}'] = rut
                        
                    
            b += -1
        return dict_perm   
    
    def state_update_pos(self,myop=False,train=True):
        interval = self.epoch_interval
        start_value = self.total_duration

        actions = self.matching_ADP(myopic=myop,training=train)
        self.action = actions
        self.historic_actions.append((self.time,self.action))
        for action in actions:
            veh = action[0]
            rout = action[1]
            if self.vehicles[veh].type == 'Private':
                self.update_priv(veh,rout)
            if self.vehicles[veh].type == 'External':
                self.update_ext(veh)
       
        self.immediate_cost = self.total_duration - start_value
        return self
    
    def state_update_pre(self,start,scenarios,realization):
        interval = self.epoch_interval
        end = start + interval
        self.time += interval
        self.immediate_cost = 0
        for centre_index, centre_value in self.centres.items():
            temporary = [i for i in scenarios[realization, centre_index].interval_filter(start,end)]
            to_remove = []
            vehicles_in_transit = [ind for ind,veh in self.vehicles.items() if veh.vlab != 0 and veh.vcent[centre_index] < 1440 and veh.type=='Private']


            for t in temporary:
                for v in vehicles_in_transit:
                    if t.remain_to_arrive < self.vehicles[v].vcent[centre_index]:
                        to_remove.append(t)
            temporary = [t.remain_to_expire for t in temporary if t not in to_remove]
            self.centres[centre_index].req += len(temporary)
            if len(temporary) > 0:
                self.centres[centre_index].crit = min(self.centres[centre_index].crit,min(temporary))
        self.action = []

        vehicles_in_transit = [ind for ind,veh in self.vehicles.items() if veh.vlab != 0 and veh.type == 'Private']
        for v in vehicles_in_transit:
            self.vehicles[v].vlab = max(0,self.vehicles[v].vlab - interval)

            for centre_index, centre_value in self.centres.items():
                self.vehicles[v].vcent[centre_index] = ((self.vehicles[v].vcent[centre_index] - interval) if (self.vehicles[v].vcent[centre_index] - interval) > 0 else 1440) if self.vehicles[v].vcent[centre_index] != 1440 else 1440

        for centre_index, centre_value in self.centres.items():
            if self.centres[centre_index].req > 0:
                if self.centres[centre_index].crit - interval < 0:
                    self.centres[centre_index].crit = 1440
                else: 
                    self.centres[centre_index].crit += - interval
                    if self.centres[centre_index].crit > (self.finishing_workshift - self.time):
                        self.centres[centre_index].crit = self.finishing_workshift - self.time
        return self

    def feature_try(self,vehicle):
        time = np.array([self.time])
        vehicles_in_lab = np.array([len([i for i in self.vehicles.values() if i.vlab ==0 and i.type=='Private'])])
        vehicles_in_route = np.array([len([i for i in self.vehicles.values() if i.vlab !=0 and i.type=='Private'])])
        crit_dead_binary = np.array([1 if len([i.crit for i in self.centres.values() if i.crit < 1440]) > 0 else 0])
        avg_crit_deadline = np.array([np.mean([i.crit for i in self.centres.values() if i.crit < 1440]) if len([i.crit for i in self.centres.values() if i.crit < 1440])>0 else 0])
        centres_optional, centres_critical = self.centres_req()
        len_centres_optional = np.array([len(centres_optional)])
        len_centres_critical = np.array([len(centres_critical)])
        ext_acum = np.array([self.external_used])

        list_try = self.historic_actions
        full_list = []
        center_mapping = {j:i for i,j in enumerate(self.centres.keys())}
        num_centres = len(self.centres)

        array_try = np.array([])
        for i in list_try:
            for j in i[1]:
                if j[0] == vehicle:
                    # Filter out 'lab' and sort centers by their order
                    filtered_route = {k: v for k, v in j[1].items() if k != 'lab'}
                    order_array = np.zeros(num_centres, dtype=int)
                    # Iterate through the filtered dictionary and fill the order_array
                    for center, value in filtered_route.items():
                        index = center_mapping[center]
                        order_array[index] = sorted(filtered_route.values()).index(value) + 1
                    full_list.append(order_array)
                else:
                    order_array = np.zeros(num_centres, dtype=int)
                    full_list.append(order_array)
        full_list = np.array([full_list])
        return time,vehicles_in_lab,vehicles_in_route,crit_dead_binary,avg_crit_deadline,len_centres_optional,len_centres_critical,ext_acum, full_list

    def feature_nn(self,vehicle):
        time = self.time
        avg_crit_deadline = np.mean([i.crit for i in self.centres.values() if i.crit < 1440]) if len([i.crit for i in self.centres.values() if i.crit < 1440])>0 else 200
        centres_in_route = 0 if self.vehicles[vehicle].type == 'External' else len([key for key,value in self.vehicles[vehicle].vcent.items() if value != 1440])
        vehicle_var = np.hstack([time,avg_crit_deadline,centres_in_route])
        
        return vehicle_var

    def get_noise(self,action):
        # std = 1 + max(0, 500 - self.realization*10)
        std = 1
        # std = 1 + ((1000 if action == 'wait' else 1000) / self.realization)
        # return abs(np.random.normal(0, std))
        return 0
    
    def get_negatives(self):
        total = 0
        for centre in self.centres.keys():
            evaluate = self.centres[centre].crit - (self.duration[('lab', centre)] + self.duration[(centre,'lab')])
            total += evaluate if evaluate < 0 else 0
        return abs(total)

    def get_values(self,neural,vehicles=None,vehicles_at_lab_priv=None,vehicles_at_lab_ext=None,routes=None,training=True):
        self.value_function = {}

        self.value_function.update({(v,r):[((copy.deepcopy(self)).update_priv(v,self.routes()[r])).feature_nn(v),None] for v in vehicles_at_lab_priv for r in routes})
        self.value_function.update({(v,'wait'):[(copy.deepcopy(self)).feature_nn(v),None] for v in vehicles})
        self.value_function.update({(v,'dispatch'):[((copy.deepcopy(self)).update_ext(v)).feature_nn(v),None] for v in vehicles_at_lab_ext})

        # Collect all feature arrays
        features_list = []
        keys_list = []

        for key, value in self.value_function.items():
            features = value[0]  # Extract the np array of features
            if features is not None:
                # Reshape features if necessary
                features = np.expand_dims(features, axis=0) if len(features.shape) == 1 else features
                features_list.append(features)
                keys_list.append(key)

        # Stack all features into a single array for batch prediction
        features_batch = np.vstack(features_list)

        # Make batch predictions
        predictions = neural.predict(features_batch,verbose=0)

        # Update the dictionary with the predictions
        for i, key in enumerate(keys_list):
            self.value_function[key][1] = predictions[i][0]

        # Print the updated dictionary to check
        return self.value_function


    def matching_ADP(self,individual_objective=False,nnmodel=None,myopic=False,training=True):
        epoch_interval = self.epoch_interval
        centres_optional, centres_critical = self.centres_req()
        vehicles = [veh_index for veh_index,veh_obj in self.vehicles.items()]
        vehicles_at_lab_priv = [veh_index for veh_index, veh_obj in self.vehicles.items() if (veh_obj.type == 'Private' and veh_obj.vlab ==0)]
        vehicles_at_lab_ext = [veh_index for veh_index, veh_obj in self.vehicles.items() if (veh_obj.type == 'External' and veh_index.split('_', 1)[1] in centres_critical)] 
        vehicles_in_route_priv = [veh_index for veh_index, veh_obj in self.vehicles.items() if (veh_obj.type == 'Private' and veh_obj.vlab !=0)]
        routes = [i for i,j in self.routes().items()]
        
        self.time


        # Initialize nnmodel only once and reuse it
        if state.nnmodel is None:
            centres_len = len(self.centres)
            state.nnmodel = nadp.NeurADP(centres_len)

        neuradp_model = state.nnmodel.model
        neuradp_target = state.nnmodel.target
        
        with gp.Model(env=env) as m:
            m.setParam('OutputFlag', 0)
            # Formulate problem

            # Create variables
            action_priv = m.addVars(vehicles_at_lab_priv,routes,vtype=GRB.BINARY,name='action_priv')
            action_ext = m.addVars(vehicles_at_lab_ext,vtype=GRB.BINARY,name='action_ext')
            wait = m.addVars(vehicles,vtype=GRB.BINARY,name='wait')
            
            # Get value function
            value_function = {} if myopic else self.get_values(neural=neuradp_target,vehicles=vehicles,vehicles_at_lab_priv=vehicles_at_lab_priv,vehicles_at_lab_ext=vehicles_at_lab_ext,routes=routes)
            # print(f'value_function is {value_function}')
            
            
            # Set Objective Function
            obj = gp.LinExpr()


            for vehicle in vehicles:
                if vehicle not in vehicles_in_route_priv:
                    # this is to the decision to do nothing
                    obj += ((0 if myopic else value_function[vehicle,'wait'][1]) + ((0 if individual_objective else (self.get_noise('wait')if training else 0))))*wait[vehicle] 

                    if vehicle in vehicles_at_lab_priv:
                        for route in routes:
                            obj += (self.routes()[route]['lab'] + (0 if myopic else value_function[vehicle,route][1]) + ((0 if individual_objective else (self.get_noise(route) if training else 0))))*action_priv[vehicle,route] 
                    
                    if vehicle in vehicles_at_lab_ext:
                            c = vehicle.split('_', 1)[1]
                            obj += ((self.duration['lab',c] + self.duration[c,'lab'])*2 + (0 if myopic else value_function[vehicle,'dispatch'][1]) + ((0 if individual_objective else (self.get_noise('dispatch') if training else 0))))*action_ext[vehicle] 
            m.setObjective(obj, GRB.MINIMIZE)
            
            # print(obj)

            # Set constraints 

            c1 = m.addConstrs((action_priv.sum(vehicle,routes) + wait[vehicle] == 1 for vehicle in vehicles_at_lab_priv),name='c1')
            c2 = m.addConstrs((action_ext.sum(vehicle) + wait[vehicle] == 1 for vehicle in vehicles_at_lab_ext),name='c2')

            for cent in centres_optional:
                routes_optional = [route for route in routes if cent in self.routes()[route].keys()]
                prinv_const = m.addConstr((action_priv.sum(vehicles_at_lab_priv,routes_optional) <= 1))
            for cent in centres_critical:
                veh_ext = [vehicle_crit for vehicle_crit in vehicles_at_lab_ext if vehicle_crit.split('_', 1)[1] == cent]
                routes_critical = [route for route in routes if cent in self.routes()[route].keys()]
                crit_const = m.addConstr((action_ext.sum(veh_ext) +  action_priv.sum(vehicles_at_lab_priv,routes_critical) == 1))
            
            m.optimize()
            m.write('modeldebug.lp')  # Saves the model in LP format

            # Print solution
            action_veh = []
            action_veh_obj = []
            if m.Status == GRB.OPTIMAL:
                solution_priv = m.getAttr('X',action_priv)
                solution_ext = m.getAttr('X',action_ext)
                waiting_decision = m.getAttr('X',wait)

                solution_priv_obj = m.getAttr('Obj',action_priv)               
                solution_ext_obj = m.getAttr('Obj',action_ext)
                waiting_decision_obj = m.getAttr('Obj',wait)
                
                for var in action_priv.values():
                    coeff = m.getObjective().getCoeff(var.index) if var.X > 0 else 0

                for vehicle in vehicles_at_lab_priv:
                    if waiting_decision[vehicle] > 0:
                        action_veh_obj.append([vehicle,[],0 if myopic else value_function[vehicle,'wait'][1]])
                    for route in routes:
                        if solution_priv[vehicle,route] > 0:
                            action_veh.append([vehicle,self.routes()[route]])
                            action_veh_obj.append([vehicle,self.routes()[route],self.routes()[route]['lab']+ (0 if myopic else value_function[vehicle,route][1])])
                
                for vehicle in vehicles_at_lab_ext:
                    if waiting_decision[vehicle] > 0:
                        action_veh_obj.append([vehicle,[],0 if myopic else value_function[vehicle,'wait'][1]])
                    if solution_ext[vehicle] > 0:
                        action_veh.append([vehicle,'dispatch'])
                        c = vehicle.split('_', 1)[1]
                        action_veh_obj.append([vehicle,'dispatch',(self.duration['lab',c] + self.duration[c,'lab'])*2 + (0 if myopic else value_function[vehicle,'dispatch'][1])])
                
        if individual_objective == False:
            return action_veh
        else:
            return action_veh_obj
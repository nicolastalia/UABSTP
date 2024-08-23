from geopy.geocoders import Nominatim
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
import matplotlib.pyplot as plt
import random
import math
import operator

os.environ['GRB_LICENSE_FILE'] = 'c9697abb-f79e-46a9-b050-2895ccc98a02'
env = gp.Env(empty=True)
env.start()


def mip_upi(veh,cent,duration,scenarios,realization):
    vehicles = [v for v in range(len(veh))]
    routes = [r for r in range(5)]
    centres = [int(index.split('_')[1]) for index,value in cent.items()]
    nodes = centres.copy()
    nodes.insert(0, 0)

    lab_dept = max(centres) + 1
    nodes.append(lab_dept)

    
    # New dictionary to hold modified keys
    dict = {}

    # Iterate over the original dictionary items
    for (key1, key2), value in duration.items():
        # Replace 'lab' with 'lab_arr' in the keys
        new_key1 = 0 if key1 == 'lab' else int(key1.split('_')[1])
        new_key2 = lab_dept if key2 == 'lab' else int(key2.split('_')[1])
        # Add the new key-value pair to the modified dictionary
        dict[(new_key1, new_key2)] = value

    arcs, duration = gp.multidict(dict)

    req1 = {}
    scen = realization
    for cent_index,cent_value in cent.items():
        for index,s in enumerate(scenarios[scen,cent_index].samples):
            req1[int(cent_index.split('_')[1]),index] = (s.arrival,s.expiration)

    req,tw_lower,tw_upper = gp.multidict(req1)

    with gp.Model(env=env) as m:
            m.setParam('TimeLimit', 240)
            m.setParam('OutputFlag', 0)
            # Formulate problem
            # Create variables
            travel = m.addVars(vehicles,routes,arcs,vtype=GRB.BINARY,name='travel')
            request = m.addVars(req,vehicles,routes,vtype=GRB.BINARY,name='request')
            arrival = m.addVars(vehicles,routes,nodes,vtype=GRB.CONTINUOUS,name='arrival')

            # Set Objective Function

            obj = gp.LinExpr()

            # Minimizing sum of the total travelling time
            for i,j in arcs:
                for v in vehicles:
                    for r in routes:
                        obj += travel[v,r,i,j]*duration[i,j]
            
            m.setObjective(obj, GRB.MINIMIZE)  

            # Constraints
            c1 = m.addConstrs((request.sum(c,k,vehicles,routes) == 1 for c,k in req),name='c1')
            c2 = m.addConstrs((arrival[v,r,nodes[-1]] <= tw_upper[i,j] + (1 - request[i,j,v,r])*3000 for v in vehicles for r in routes for i,j in req),name='c2')
            c3 = m.addConstrs((arrival[v,r,k] >= tw_lower[i,j] - (1 - request[i,j,v,r])*3000 for v in vehicles for r in routes for k in centres for i,j in req),name='c3')
            c4 = m.addConstrs((arrival[v,r,j] >= duration[i,j] + arrival[v,r,i] - (1- travel[v,r,i,j])*2000 for i,j in arcs for v in vehicles for r in routes),name='c4')
            c5 = m.addConstrs((travel.sum(v,r,nodes[0],centres) == travel.sum(v,r,centres,nodes[-1]) for v in vehicles for r in routes),name='c5')
            c6 = m.addConstrs((request[j,re,v,r] <= travel.sum(v,r,nodes[:-1],j) for j,re in req for v in vehicles for r in routes),name='c6')
            c7 = m.addConstrs((travel[v,r,0,j] >= travel.sum(v,r,j,nodes[:-1]) for j in centres for v in vehicles for r in routes),name='c7')
            c8 = m.addConstrs((travel.sum(v,r,nodes[0],centres) <= 1 for v in vehicles for r in routes),name='c8')
            c9 = m.addConstrs((travel.sum(v,r,nodes[:-1],j) == travel.sum(v,r,j,nodes[1:]) for j in nodes[1:-1] for v in vehicles for r in routes),name='c9')
            c10 = m.addConstrs((arrival[v,r+1,nodes[0]] >= arrival[v,r,nodes[-1]] for v in vehicles for r in routes[:-1]),name='c10')
            c11 = m.addConstrs((arrival[v,r+1,nodes[0]] >= arrival[v,r,nodes[0]] for v in vehicles for r in routes[:-1]),name='c11')
            c15 = m.addConstr((arrival[0,0,0] == 0),name='c15')

            m.optimize()

            objective = 0
            runtime = 0

            # m.write("2c1l.lp")
            if m.Status == GRB.OPTIMAL:
                # for v in m.getVars():
                #     if v.X > 0:
                #         print(f'{v.VarName}: {v.X}')
                print('Obj: %g' % m.ObjVal)
                objective = m.ObjVal
                runtime = m.Runtime  
                mip_gap = m.MIPGap

            else:
                # Handle cases where the solution is not optimal but still feasible
                if m.Status in [GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.ITERATION_LIMIT, GRB.NODE_LIMIT]:
                    print('Solution is not optimal.')
                    if m.SolCount > 0:
                        # Retrieve the best objective value found
                        print('Best Obj: %g' % m.ObjVal)
                        objective = m.ObjVal
                        # Retrieve the MIP gap
                        print('MIP Gap: %g' % m.MIPGap)
                        mip_gap = m.MIPGap
                        runtime = m.Runtime
                    else:
                        print('No feasible solution found.')
                        objective = 0
                        runtime = m.Runtime  
                        mip_gap = 0
                else:
                    print('Model was not solved successfully.')
                    objective = 0
                    runtime = m.Runtime  
                    mip_gap = 0
    return objective, runtime,mip_gap     
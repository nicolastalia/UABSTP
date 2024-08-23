# importing libraries
from geopy.geocoders import Nominatim
import numpy as np
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



class sample:
    def __init__(self,arrival,expiration):
        self.arrival = arrival
        self.expiration = expiration
        self.remain_to_arrive = arrival
        self.remain_to_expire = expiration
    def __str__(self):
        return f'arrival: {self.arrival}, expiration: {self.expiration}, remain_to_arrive: {self.remain_to_arrive}, remain_to_expire: {self.remain_to_expire}'

class sample_path:
    i = 0
    z = 1136
    def __init__(self,centre,scenario=1,rate=None,working_hours=None):
        self.centre = centre
        self.scenario = scenario 
        self.rate = rate

        # m = 16384
        m = 2**200
        a = 4781
        c = 8521
        

        self.rate = 1/rate #number of specimens per hour
        starting_at = 8
        self.working_hours = working_hours
        perish = 180
        

        start_time = starting_at*60
        list = []
        while start_time <= (starting_at + self.working_hours)*60:
            sample_path.i += 1
            sample_path.z = (a*sample_path.z + c) % m
            u = sample_path.z/m
            e = - np.log(1-u) / self.rate
            start_time = np.round(e + start_time,0)
            if start_time <= (starting_at + self.working_hours)*60:
                list.append(sample(start_time,start_time+perish))
        self.samples = list
        
    def restart():
        sample_path.i = 0
        sample_path.z = 1136

    def interval_filter(self,start,end):
        req = [samp for samp in self.samples if samp.arrival > start and samp.arrival <= end]
        for samp in req:
            samp.remain_to_arrive = samp.arrival - start
            samp.remain_to_expire = samp.expiration - start
        return req
    
    def request_in_interval(self,start,end):
        req = [samp for samp in self.samples if samp.arrival > start and samp.arrival <= end]
        return len(req)
    
    def critical_in_interval(self,start,end):
        req = [samp.expiration for samp in self.samples if samp.arrival > start and samp.arrival <= end]
        return min(req) - start
    
    def __str__(self):
        return f'(centre:{self.centre},scenario: {self.scenario}, arrivals: {[(arrival.arrival, arrival.expiration) for arrival in self.samples]})'

class SamplePathPreDefine:
    def __init__(self,centre,scenario=1,rate=1,start_time=480,working_hours=480,interval=10,perish=90):
        self.centre = centre
        self.scenario = scenario 
        self.rate = rate
        self.start_time = start_time
        self.working_hours = working_hours
        self.interval = interval
        self.perish = perish

        list = []
        arrival = self.start_time + rate
        while arrival <= self.start_time + self.working_hours:
            list.append(sample(arrival,arrival+perish))
            arrival += rate
        self.samples = list
    
    def interval_filter(self,start,end):
        req = [samp for samp in self.samples if samp.arrival > start and samp.arrival <= end]
        for samp in req:
            samp.remain_to_arrive = samp.arrival - start
            samp.remain_to_expire = samp.expiration - start
        return req

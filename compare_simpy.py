"""
Validation Test: Custom Engine vs. SimPy
----------------------------------------
This script runs a head-to-Head comparison between:
1. Your Custom NetworkSimulator (Event-Scheduling)
2. SimPy (Process-Interaction / Standard Library)

Scenario: M/M/1 Queue
- Arrival Rate (lambda): 4.0
- Service Rate (mu): 5.0
- Theoretical Avg Wait (Wq): 0.80 seconds
- Duration: 50,000 seconds (to ensure convergence)
"""

import simpy
import random
import heapq
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

# ==========================================
# PART 1: YOUR CUSTOM SIMULATION ENGINE
# (Copied from your simulation_engine.py)
# ==========================================

class EventType(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"

class DistributionType(Enum):
    EXPONENTIAL = "Exponential"

@dataclass(order=True)
class Event:
    time: float
    event_type: EventType = field(compare=False)
    packet_id: int = field(compare=False, default=-1)
    server_id: int = field(compare=False, default=0)

@dataclass
class Packet:
    packet_id: int
    arrival_time: float
    service_start_time: Optional[float] = None
    departure_time: Optional[float] = None
    service_time: float = 0.0
    
    @property
    def waiting_time(self) -> float:
        return self.service_start_time - self.arrival_time if self.service_start_time else 0.0

@dataclass
class Server:
    server_id: int
    busy: bool = False
    current_packet: Optional[Packet] = None

class CustomSimulator:
    def __init__(self, arr_rate, svc_rate, max_time):
        self.arr_rate = arr_rate
        self.svc_rate = svc_rate
        self.max_time = max_time
        
        self.clock = 0.0
        self.queue = []
        self.server = Server(0)
        self.event_list = []
        self.completed_packets = []
        self.packet_counter = 0
        
        # Use Python's random to be fair with SimPy
        random.seed(42) 
        
        # Schedule first arrival
        self.schedule(Event(0.0, EventType.ARRIVAL, 0))

    def schedule(self, event):
        heapq.heappush(self.event_list, event)

    def run(self):
        while self.event_list:
            event = heapq.heappop(self.event_list)
            self.clock = event.time
            if self.clock > self.max_time: break
            
            if event.event_type == EventType.ARRIVAL:
                self.handle_arrival(event)
            else:
                self.handle_departure(event)
                
        return self.get_stats()

    def handle_arrival(self, event):
        p = Packet(event.packet_id, self.clock)
        
        if not self.server.busy:
            self.server.busy = True
            self.server.current_packet = p
            p.service_start_time = self.clock
            svc_time = random.expovariate(self.svc_rate)
            self.schedule(Event(self.clock + svc_time, EventType.DEPARTURE, p.packet_id))
        else:
            self.queue.append(p)
            
        # Schedule next arrival
        next_arr = random.expovariate(self.arr_rate)
        self.packet_counter += 1
        self.schedule(Event(self.clock + next_arr, EventType.ARRIVAL, self.packet_counter))

    def handle_departure(self, event):
        # Finish current
        p = self.server.current_packet
        p.departure_time = self.clock
        self.completed_packets.append(p)
        
        if self.queue:
            next_p = self.queue.pop(0)
            self.server.current_packet = next_p
            next_p.service_start_time = self.clock
            svc_time = random.expovariate(self.svc_rate)
            self.schedule(Event(self.clock + svc_time, EventType.DEPARTURE, next_p.packet_id))
        else:
            self.server.busy = False
            self.server.current_packet = None

    def get_stats(self):
        waits = [p.waiting_time for p in self.completed_packets]
        return np.mean(waits) if waits else 0.0

# ==========================================
# PART 2: SIMPY IMPLEMENTATION
# ==========================================

class SimPyModel:
    def __init__(self, arr_rate, svc_rate, max_time):
        self.env = simpy.Environment()
        self.server = simpy.Resource(self.env, capacity=1)
        self.arr_rate = arr_rate
        self.svc_rate = svc_rate
        self.max_time = max_time
        self.wait_times = []
        
        # Use same seed to be fair
        random.seed(42) 

    def packet_generator(self):
        i = 0
        while True:
            yield self.env.timeout(random.expovariate(self.arr_rate))
            i += 1
            self.env.process(self.packet_process(f'Packet {i}'))

    def packet_process(self, name):
        arrival_time = self.env.now
        
        # Request Server
        with self.server.request() as request:
            yield request # Wait in queue
            
            # Service Start
            wait = self.env.now - arrival_time
            self.wait_times.append(wait)
            
            # Service Duration
            yield self.env.timeout(random.expovariate(self.svc_rate))

    def run(self):
        self.env.process(self.packet_generator())
        self.env.run(until=self.max_time)
        return np.mean(self.wait_times)

# ==========================================
# PART 3: HEAD-TO-HEAD BATTLE
# ==========================================

if __name__ == "__main__":
    LAMBDA = 4.0
    MU = 5.0
    TIME = 50000.0  # Long simulation for convergence
    
    print(f"--- SIMULATION CONFIGURATION ---")
    print(f"Arrival Rate (λ): {LAMBDA}")
    print(f"Service Rate (μ): {MU}")
    print(f"Time Horizon:     {TIME} seconds")
    print("-" * 40)

    # 1. Theoretical Result
    rho = LAMBDA/MU
    theo_wq = LAMBDA / (MU * (MU - LAMBDA)) # Erlang Formula for Wq
    print(f"🏆 THEORETICAL Target (Wq): {theo_wq:.5f} s")
    print("-" * 40)

    # 2. Run Custom
    print("Running Custom Simulator...")
    custom_sim = CustomSimulator(LAMBDA, MU, TIME)
    custom_wq = custom_sim.run()
    
    # 3. Run SimPy
    print("Running SimPy Model...")
    simpy_model = SimPyModel(LAMBDA, MU, TIME)
    simpy_wq = simpy_model.run()

    # 4. Results
    print("\n--- FINAL RESULTS ---")
    print(f"Custom Engine Wq: {custom_wq:.5f} s  (Error: {abs(custom_wq-theo_wq)/theo_wq*100:.2f}%)")
    print(f"SimPy Library Wq: {simpy_wq:.5f} s  (Error: {abs(simpy_wq-theo_wq)/theo_wq*100:.2f}%)")
    
    diff = abs(custom_wq - simpy_wq)
    print(f"\nDifference between Engines: {diff:.6f} s")
    
    if diff < 0.05:
        print("\n✅ SUCCESS: Your custom engine matches SimPy accuracy!")
    else:
        print("\n⚠️ WARNING: Significant divergence detected.")

"""
Network Traffic Simulation Engine
Merged Version: Original Features + SWE 627 Requirements
Features: M/M/c, LCG RNG, Inverse Transform, Warm-up/Data Deletion, Chi-Square
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import pandas as pd
from scipy import stats as scipy_stats
import math
import json

class EventType(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"
    WARMUP_END = "WARMUP_END"  # New: For Data Deletion

class DistributionType(Enum):
    EXPONENTIAL = "Exponential"
    NORMAL = "Normal"
    UNIFORM = "Uniform"
    WEIBULL = "Weibull"
    GAMMA = "Gamma"
    LOGNORMAL = "Log-normal"
    POISSON = "Poisson"

# --- NEW: Course Requirement - Linear Congruential Generator ---
class LCG:
    """
    Linear Congruential Generator (LCG)
    Implements X_{i} = (a * X_{i-1} + c) mod m
    """
    def __init__(self, seed: int, a: int = 16807, c: int = 0, m: int = 2147483647):
        # Default a=7^5, m=2^31-1 (Lewis, Goodman, and Miller, 1969)
        self.state = seed if seed != 0 else 1 
        self.a = a
        self.c = c
        self.m = m

    def random(self) -> float:
        """Returns a Uniform(0,1) float"""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

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
    server_id: int = 0
    
    @property
    def waiting_time(self) -> Optional[float]:
        if self.service_start_time is not None:
            return self.service_start_time - self.arrival_time
        return None
    
    @property
    def system_time(self) -> Optional[float]:
        if self.departure_time is not None:
            return self.departure_time - self.arrival_time
        return None

@dataclass
class Server:
    server_id: int
    busy: bool = False
    current_packet: Optional[Packet] = None
    total_busy_time: float = 0.0
    last_busy_start: float = 0.0
    packets_served: int = 0

@dataclass
class SimulationConfig:
    # --- New: RNG & Warmup Controls ---
    use_lcg: bool = False
    warmup_time: float = 0.0
    
    # Arrival parameters
    arrival_distribution: DistributionType = DistributionType.EXPONENTIAL
    arrival_rate: float = 5.0
    arrival_mean: float = 0.2
    arrival_std: float = 0.05
    arrival_min: float = 0.1
    arrival_max: float = 0.3
    arrival_shape: float = 2.0 
    arrival_scale: float = 0.2 
    
    # Service parameters
    service_distribution: DistributionType = DistributionType.EXPONENTIAL
    service_rate: float = 8.0
    service_mean: float = 0.125
    service_std: float = 0.03
    service_min: float = 0.05
    service_max: float = 0.2
    service_shape: float = 2.0
    service_scale: float = 0.125
    
    # System parameters
    num_servers: int = 1
    queue_capacity: int = 100
    simulation_time: float = 100.0
    random_seed: Optional[int] = 42
    
    def get_traffic_intensity(self) -> float:
        """Calculate traffic intensity (rho)"""
        eff_arr = self.arrival_rate if self.arrival_distribution == DistributionType.EXPONENTIAL else (1/self.arrival_mean if self.arrival_mean > 0 else 0)
        eff_svc = self.service_rate if self.service_distribution == DistributionType.EXPONENTIAL else (1/self.service_mean if self.service_mean > 0 else 0)
        if eff_svc == 0 or self.num_servers == 0: return 0
        return eff_arr / (self.num_servers * eff_svc)
    
    def to_dict(self) -> Dict:
        return self.__dict__.copy()

class RandomGenerator:
    """
    Handles random variate generation.
    Switches between NumPy and Custom LCG based on config.
    Implements Inverse Transform Method for Exponential/Uniform.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.np_rng = np.random.default_rng(config.random_seed)
        if config.use_lcg:
            self.lcg = LCG(config.random_seed if config.random_seed else 12345)
    
    def _get_u01(self) -> float:
        """Helper to get U(0,1) from selected source"""
        if self.config.use_lcg:
            return self.lcg.random()
        return self.np_rng.random()

    def generate(self, dist_type: DistributionType, **params) -> float:
        # --- Course Requirement: Inverse Transform Method ---
        if dist_type == DistributionType.EXPONENTIAL:
            # Formula: x = -ln(1-u) / lambda
            u = self._get_u01()
            rate = params.get('rate', 1.0)
            return -(math.log(1.0 - u)) / rate
        
        elif dist_type == DistributionType.UNIFORM:
            # Formula: x = min + (max-min)*u
            u = self._get_u01()
            low = params.get('min', 0.0)
            high = params.get('max', 1.0)
            return low + (high - low) * u

        # --- Other Distributions (Hybrid approach) ---
        # For complex distributions, we use Numpy but source the randomness 
        # from LCG if selected, or just use Numpy's optimized generators.
        
        elif dist_type == DistributionType.NORMAL:
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            if self.config.use_lcg:
                # Box-Muller transform for LCG
                u1 = self._get_u01()
                u2 = self._get_u01()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                return max(0.001, mean + z * std)
            else:
                return max(0.001, self.np_rng.normal(mean, std))
        
        elif dist_type == DistributionType.POISSON:
            lam = params.get('lam', 1.0)
            if self.config.use_lcg:
                # Knuth's algorithm for Poisson
                L = math.exp(-lam)
                k = 0
                p = 1.0
                while p > L:
                    k += 1
                    p *= self._get_u01()
                return float(k - 1)
            else:
                return float(self.np_rng.poisson(lam))

        # Fallback to numpy for Weibull/Gamma/Lognormal to maintain original project quality
        elif dist_type == DistributionType.WEIBULL:
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            return scale * self.np_rng.weibull(shape)
        
        elif dist_type == DistributionType.GAMMA:
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            return self.np_rng.gamma(shape, scale)
            
        elif dist_type == DistributionType.LOGNORMAL:
            mean = params.get('mean', 0.0)
            sigma = params.get('std', 1.0)
            return self.np_rng.lognormal(mean, sigma)
            
        return 1.0

@dataclass
class EventLogEntry:
    event_number: int
    clock_time: float
    event_type: str
    packet_id: int
    server_id: int
    queue_length: int
    servers_busy: int
    total_servers: int

@dataclass
class SystemState:
    clock: float = 0.0
    servers: List[Server] = field(default_factory=list)
    queue: List[Packet] = field(default_factory=list)
    
    # Accumulators
    total_arrivals: int = 0
    total_departures: int = 0
    total_drops: int = 0
    
    # Time-weighted stats
    area_under_queue: float = 0.0
    area_under_system: float = 0.0
    last_event_time: float = 0.0
    
    completed_packets: List[Packet] = field(default_factory=list)
    
    # --- New: Warmup flag ---
    in_warmup: bool = False

    def queue_length(self) -> int: return len(self.queue)
    def busy_servers(self) -> int: return sum(1 for s in self.servers if s.busy)
    def packets_in_system(self) -> int: return len(self.queue) + self.busy_servers()
    def all_servers_busy(self) -> bool: return all(s.busy for s in self.servers)
    
    def get_idle_server(self) -> Optional[Server]:
        for s in self.servers:
            if not s.busy: return s
        return None

class NetworkSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = RandomGenerator(config)
        self.state = SystemState()
        self.state.in_warmup = (config.warmup_time > 0)
        self.event_list: List[Event] = []
        self.event_log: List[EventLogEntry] = []
        self.packet_counter = 0
        self.event_counter = 0
        self.time_series: List[Dict] = []
        
    def schedule_event(self, event: Event):
        heapq.heappush(self.event_list, event)
    
    def get_next_event(self) -> Optional[Event]:
        if self.event_list: return heapq.heappop(self.event_list)
        return None
        
    def update_statistics(self, new_time: float):
        # Don't record area stats during warmup if we are essentially ignoring that period
        # But for 'reset' logic, we usually track then clear. 
        # Here we just track normally, and handle_warmup_end will clear them.
        time_delta = new_time - self.state.last_event_time
        if not self.state.in_warmup:
            self.state.area_under_queue += self.state.queue_length() * time_delta
            self.state.area_under_system += self.state.packets_in_system() * time_delta
        self.state.last_event_time = new_time

    def handle_warmup_end(self):
        """Reset statistics after warmup period (T0)"""
        self.state.total_arrivals = 0
        self.state.total_departures = 0
        self.state.total_drops = 0
        self.state.area_under_queue = 0.0
        self.state.area_under_system = 0.0
        self.state.completed_packets = []
        
        # Reset server busy times
        for s in self.state.servers:
            s.total_busy_time = 0.0
            s.packets_served = 0
            # If busy, reset busy start to now so future calc is correct
            if s.busy:
                s.last_busy_start = self.state.clock
                
        self.state.in_warmup = False
        # Log the reset
        self.log_event("WARMUP_RESET", -1, -1)

    def log_event(self, event_type: str, packet_id: int, server_id: int):
        self.event_counter += 1
        # Store log
        entry = EventLogEntry(
            self.event_counter, self.state.clock, event_type, packet_id, server_id,
            self.state.queue_length(), self.state.busy_servers(), self.config.num_servers
        )
        self.event_log.append(entry)
        
        # Store time series for plotting
        if not self.state.in_warmup:
            self.time_series.append({
                'time': self.state.clock,
                'queue_length': self.state.queue_length(),
                'packets_in_system': self.state.packets_in_system(),
                'servers_busy': self.state.busy_servers()
            })

    def run(self) -> Dict:
        # Initialize
        self.state = SystemState()
        self.state.servers = [Server(i) for i in range(self.config.num_servers)]
        self.state.in_warmup = (self.config.warmup_time > 0)
        self.event_list = []
        self.event_log = []
        self.time_series = []
        self.packet_counter = 0
        self.event_counter = 0
        
        # Schedule first arrival
        self.packet_counter += 1
        self.schedule_event(Event(0.0, EventType.ARRIVAL, self.packet_counter))
        
        # Schedule Warmup End if needed
        if self.config.warmup_time > 0:
            self.schedule_event(Event(self.config.warmup_time, EventType.WARMUP_END))
            
        while self.event_list:
            event = self.get_next_event()
            if event.time > self.config.simulation_time:
                break
            
            self.update_statistics(event.time)
            self.state.clock = event.time
            
            if event.event_type == EventType.ARRIVAL:
                self.handle_arrival(event)
            elif event.event_type == EventType.DEPARTURE:
                self.handle_departure(event)
            elif event.event_type == EventType.WARMUP_END:
                self.handle_warmup_end()
                
        # Final update
        self.update_statistics(self.config.simulation_time)
        return self.compute_statistics()

    def handle_arrival(self, event):
        if not self.state.in_warmup:
            self.state.total_arrivals += 1
            
        packet = Packet(event.packet_id, event.time)
        
        # Logic: Queue Capacity
        if self.config.queue_capacity > 0 and \
           self.state.queue_length() >= self.config.queue_capacity and \
           self.state.all_servers_busy():
            if not self.state.in_warmup:
                self.state.total_drops += 1
            self.log_event("DROP", event.packet_id, -1)
        else:
            idle_server = self.state.get_idle_server()
            if idle_server:
                # Start Service
                idle_server.busy = True
                idle_server.current_packet = packet
                idle_server.last_busy_start = event.time
                packet.service_start_time = event.time
                packet.server_id = idle_server.server_id
                
                # Generate Service Time
                svc_time = self.generate_service_time()
                packet.service_time = svc_time
                
                self.schedule_event(Event(event.time + svc_time, EventType.DEPARTURE, 
                                          packet.packet_id, idle_server.server_id))
                self.log_event("ARRIVAL", event.packet_id, idle_server.server_id)
            else:
                # Enqueue
                self.state.queue.append(packet)
                self.log_event("ARRIVAL", event.packet_id, -1)
                
        # Schedule Next Arrival
        inter_val = self.generate_interarrival_time()
        next_time = event.time + inter_val
        if next_time <= self.config.simulation_time:
            self.packet_counter += 1
            self.schedule_event(Event(next_time, EventType.ARRIVAL, self.packet_counter))

    def handle_departure(self, event):
        server = self.state.servers[event.server_id]
        packet = server.current_packet
        
        if not self.state.in_warmup:
            self.state.total_departures += 1
            if packet:
                packet.departure_time = event.time
                self.state.completed_packets.append(packet)
                server.packets_served += 1
                server.total_busy_time += (event.time - server.last_busy_start)
        
        self.log_event("DEPARTURE", event.packet_id, server.server_id)
        
        if self.state.queue:
            next_packet = self.state.queue.pop(0)
            server.current_packet = next_packet
            next_packet.service_start_time = event.time
            next_packet.server_id = server.server_id
            server.last_busy_start = event.time
            
            svc_time = self.generate_service_time()
            next_packet.service_time = svc_time
            
            self.schedule_event(Event(event.time + svc_time, EventType.DEPARTURE,
                                      next_packet.packet_id, server.server_id))
        else:
            server.busy = False
            server.current_packet = None

    def generate_interarrival_time(self) -> float:
        # Wrapper to pass parameters cleanly
        c = self.config
        return self.rng.generate(c.arrival_distribution, 
                                 rate=c.arrival_rate, mean=c.arrival_mean, std=c.arrival_std,
                                 min=c.arrival_min, max=c.arrival_max, shape=c.arrival_shape,
                                 scale=c.arrival_scale, lam=c.arrival_mean)

    def generate_service_time(self) -> float:
        c = self.config
        return self.rng.generate(c.service_distribution,
                                 rate=c.service_rate, mean=c.service_mean, std=c.service_std,
                                 min=c.service_min, max=c.service_max, shape=c.service_shape,
                                 scale=c.service_scale, lam=c.service_mean)

    def compute_statistics(self) -> Dict:
        # Compute final stats (similar to original code)
        sim_duration = self.config.simulation_time - self.config.warmup_time
        if sim_duration <= 0: sim_duration = 1.0
        
        waiting_times = [p.waiting_time for p in self.state.completed_packets if p.waiting_time is not None]
        system_times = [p.system_time for p in self.state.completed_packets if p.system_time is not None]
        
        # Server util
        total_busy = sum(s.total_busy_time for s in self.state.servers)
        # Add ongoing busy time for stats? Usually handled by update_statistics logic for area, 
        # but for direct server busy time we need to account for end of sim.
        # Simplified for robustness:
        utilization = total_busy / (self.config.num_servers * sim_duration)
        
        return {
            'total_arrivals': self.state.total_arrivals,
            'total_departures': self.state.total_departures,
            'total_drops': self.state.total_drops,
            'average_queue_length': self.state.area_under_queue / sim_duration,
            'average_system_length': self.state.area_under_system / sim_duration,
            'average_waiting_time': np.mean(waiting_times) if waiting_times else 0,
            'average_system_time': np.mean(system_times) if system_times else 0,
            'server_utilization': utilization,
            'throughput': self.state.total_departures / sim_duration,
            'drop_rate': self.state.total_drops / self.state.total_arrivals if self.state.total_arrivals > 0 else 0,
            'waiting_times': waiting_times, # For histograms
            'system_times': system_times
        }

    # -- Export Helpers --
    def get_event_log_dataframe(self) -> pd.DataFrame:
        if not self.event_log: return pd.DataFrame()
        return pd.DataFrame([vars(e) for e in self.event_log])
    
    def get_time_series_dataframe(self) -> pd.DataFrame:
        if not self.time_series: return pd.DataFrame()
        return pd.DataFrame(self.time_series)

# --- New: Input Analysis Logic (Chi-Square) ---
def perform_chi_square_test(observed_data: List[float], dist_type: str, mean: float) -> Dict:
    n = len(observed_data)
    if n < 5: return {'error': 'Insufficient data'}
    
    k = max(5, int(np.sqrt(n))) 
    sorted_data = sorted(observed_data)
    expected_freq = n / k
    
    # Calculate intervals based on theoretical CDF to get equal probabilities
    bin_edges = []
    for i in range(k + 1):
        p = i / k
        if dist_type == "Exponential":
            if p >= 1.0: val = float('inf')
            else: val = -mean * math.log(1.0 - p)
        else:
            val = sorted_data[-1] * p # Fallback for linear
        bin_edges.append(val)
        
    observed_freqs = [0] * k
    current_bin = 0
    for x in sorted_data:
        while current_bin < k and x > bin_edges[current_bin+1]:
            current_bin += 1
        if current_bin < k:
            observed_freqs[current_bin] += 1
            
    chi_sq = sum(((o - expected_freq) ** 2) / expected_freq for o in observed_freqs)
    df = k - 1 - 1 # k - s - 1
    crit = scipy_stats.chi2.ppf(0.95, df)
    p_val = 1 - scipy_stats.chi2.cdf(chi_sq, df)
    
    return {
        'chi2': chi_sq, 'critical': crit, 'p_value': p_val, 
        'reject': chi_sq > crit, 'observed': observed_freqs, 'expected': expected_freq
    }

# --- Theoretical Helpers ---
def compute_mmc_theoretical(lam, mu, c):
    # (Same as original code)
    rho = lam / (c * mu)
    if rho >= 1: return {'stable': False, 'rho': rho}
    a = lam/mu
    sum_terms = sum((a**n)/math.factorial(n) for n in range(c))
    last = (a**c)/(math.factorial(c)*(1-rho))
    P0 = 1/(sum_terms + last)
    Lq = (P0 * (a**c) * rho) / (math.factorial(c) * (1-rho)**2)
    return {'stable': True, 'rho': rho, 'Lq': Lq, 'Wq': Lq/lam}

def compute_confidence_interval(data, confidence=0.95):
    if len(data) < 2: return 0, 0, 0
    n = len(data)
    m = np.mean(data)
    se = scipy_stats.sem(data)
    h = se * scipy_stats.t.ppf((1+confidence)/2, n-1)
    return m, m-h, m+h

def run_replications(config, num_reps=10):
    results = {'Wq': [], 'Lq': []}
    base_seed = config.random_seed
    for i in range(num_reps):
        config.random_seed = base_seed + i
        sim = NetworkSimulator(config)
        res = sim.run()
        results['Wq'].append(res['average_waiting_time'])
        results['Lq'].append(res['average_queue_length'])
    return results

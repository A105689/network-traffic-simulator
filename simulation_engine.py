"""
Network Traffic Simulation Engine (Educational Edition)
Features: 
- LCG RNG (Linear Congruential Generator) for reproducibility
- Inverse Transform Method for variate generation
- Warm-up/Data Deletion support
- Comparative Analysis Tools
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import pandas as pd
from scipy import stats as scipy_stats
import math
import json

class EventType(Enum):
    """Types of events in the network simulation"""
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"
    WARMUP_END = "WARMUP_END"

class DistributionType(Enum):
    """Supported probability distributions"""
    EXPONENTIAL = "Exponential"
    NORMAL = "Normal"
    UNIFORM = "Uniform"
    POISSON = "Poisson"

# --- CORE COMPONENT: Linear Congruential Generator ---
class LCG:
    """
    Linear Congruential Generator (LCG)
    Equation: X_{i} = (a * X_{i-1} + c) mod m
    """
    def __init__(self, seed: int, a: int = 16807, c: int = 0, m: int = 2147483647):
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
    """Event class for the priority queue (FEL)"""
    time: float
    event_type: EventType = field(compare=False)
    packet_id: int = field(compare=False, default=-1)
    server_id: int = field(compare=False, default=0)
    
    def __repr__(self):
        return f"Event({self.time:.4f}, {self.event_type.value}, P{self.packet_id}, S{self.server_id})"

@dataclass
class Packet:
    """Represents a network packet with full lifecycle tracking"""
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
    """Represents a server in the M/M/c system"""
    server_id: int
    busy: bool = False
    current_packet: Optional[Packet] = None
    total_busy_time: float = 0.0
    last_busy_start: float = 0.0
    packets_served: int = 0

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""
    # RNG & Warmup
    use_lcg: bool = False
    lcg_a: int = 16807      # Multiplier
    lcg_c: int = 0          # Increment
    lcg_m: int = 2147483647 # Modulus
    warmup_time: float = 0.0
    random_seed: Optional[int] = 42
    
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

    def get_traffic_intensity(self) -> float:
        """Calculate traffic intensity (rho)"""
        if self.arrival_distribution == DistributionType.EXPONENTIAL:
            eff_arr = self.arrival_rate
        else:
            eff_arr = 1 / self.arrival_mean if self.arrival_mean > 0 else 0
            
        if self.service_distribution == DistributionType.EXPONENTIAL:
            eff_svc = self.service_rate
        else:
            eff_svc = 1 / self.service_mean if self.service_mean > 0 else 0
            
        if eff_svc == 0 or self.num_servers == 0: return 0
        return eff_arr / (self.num_servers * eff_svc)

    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class RandomGenerator:
    """
    Handles random variate generation.
    Supports both standard NumPy and custom LCG with Inverse Transform.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.np_rng = np.random.default_rng(config.random_seed)
        if config.use_lcg:
            self.lcg = LCG(
                seed=config.random_seed if config.random_seed else 12345,
                a=config.lcg_a,
                c=config.lcg_c,
                m=config.lcg_m
            )
    
    def _get_u01(self) -> float:
        """Get Uniform(0,1) from selected source"""
        if self.config.use_lcg:
            return self.lcg.random()
        return self.np_rng.random()

    def generate(self, dist_type: DistributionType, **params) -> float:
        """Generate random variate"""
        # --- COURSE REQUIREMENT: INVERSE TRANSFORM METHOD ---
        if dist_type == DistributionType.EXPONENTIAL:
            # X = -ln(1-U) / lambda
            u = self._get_u01()
            rate = params.get('rate', 1.0)
            return -(math.log(1.0 - u)) / rate
        
        elif dist_type == DistributionType.UNIFORM:
            # X = min + (max-min)*U
            u = self._get_u01()
            low = params.get('min', 0.0)
            high = params.get('max', 1.0)
            return low + (high - low) * u

        # --- HYBRID APPROACH FOR COMPLEX DISTRIBUTIONS ---
        elif dist_type == DistributionType.NORMAL:
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            if self.config.use_lcg:
                # Box-Muller for LCG
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
            
        return 1.0

@dataclass
class EventLogEntry:
    event_number: int
    clock_time: float
    event_type: str
    packet_id: int
    server_id: int
    queue_length_before: int
    queue_length_after: int
    servers_busy: int
    total_servers: int
    packets_in_system: int
    cumulative_arrivals: int
    cumulative_departures: int
    cumulative_drops: int

@dataclass
class SystemState:
    clock: float = 0.0
    servers: List[Server] = field(default_factory=list)
    queue: List[Packet] = field(default_factory=list)
    total_arrivals: int = 0
    total_departures: int = 0
    total_drops: int = 0
    area_under_queue: float = 0.0
    area_under_system: float = 0.0
    last_event_time: float = 0.0
    completed_packets: List[Packet] = field(default_factory=list)
    in_warmup: bool = False

    def queue_length(self) -> int: return len(self.queue)
    def busy_servers(self) -> int: return sum(1 for s in self.servers if s.busy)
    def packets_in_system(self) -> int: return len(self.queue) + self.busy_servers()
    def all_servers_busy(self) -> bool: return all(s.busy for s in self.servers)
    
    def get_idle_server(self) -> Optional[Server]:
        for server in self.servers:
            if not server.busy: return server
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
        if self.event_list:
            return heapq.heappop(self.event_list)
        return None
        
    def update_statistics(self, new_time: float):
        time_delta = new_time - self.state.last_event_time
        self.state.area_under_queue += self.state.queue_length() * time_delta
        self.state.area_under_system += self.state.packets_in_system() * time_delta
        self.state.last_event_time = new_time

    def handle_warmup_end(self):
        self.state.total_arrivals = 0
        self.state.total_departures = 0
        self.state.total_drops = 0
        self.state.area_under_queue = 0.0
        self.state.area_under_system = 0.0
        self.state.completed_packets = []
        self.time_series = [] # Clear warmup traces
        
        for s in self.state.servers:
            s.total_busy_time = 0.0
            s.packets_served = 0
            if s.busy:
                s.last_busy_start = self.state.clock
                
        self.state.in_warmup = False
        self.log_event("WARMUP_RESET", -1, -1, 0, 0)

    def log_event(self, event_type: str, packet_id: int, server_id: int, q_before: int, q_after: int):
        self.event_counter += 1
        entry = EventLogEntry(
            event_number=self.event_counter,
            clock_time=self.state.clock,
            event_type=event_type,
            packet_id=packet_id,
            server_id=server_id,
            queue_length_before=q_before,
            queue_length_after=q_after,
            servers_busy=self.state.busy_servers(),
            total_servers=self.config.num_servers,
            packets_in_system=self.state.packets_in_system(),
            cumulative_arrivals=self.state.total_arrivals,
            cumulative_departures=self.state.total_departures,
            cumulative_drops=self.state.total_drops
        )
        self.event_log.append(entry)
        
        if not self.state.in_warmup:
            current_duration = self.state.clock - self.config.warmup_time
            running_avg_lq = 0.0
            if current_duration > 0:
                running_avg_lq = self.state.area_under_queue / current_duration

            self.time_series.append({
                'time': self.state.clock,
                'queue_length': q_after,
                'running_avg_lq': running_avg_lq,
                'packets_in_system': self.state.packets_in_system(),
                'servers_busy': self.state.busy_servers(),
                'server_utilization': self.state.busy_servers() / self.config.num_servers if self.config.num_servers > 0 else 0
            })

    def run(self) -> Dict:
        self.state = SystemState()
        self.state.servers = [Server(i) for i in range(self.config.num_servers)]
        self.state.in_warmup = (self.config.warmup_time > 0)
        self.event_list = []
        self.event_log = []
        self.time_series = []
        self.packet_counter = 0
        self.event_counter = 0
        
        self.packet_counter += 1
        self.schedule_event(Event(0.0, EventType.ARRIVAL, self.packet_counter))
        
        if self.config.warmup_time > 0:
            self.schedule_event(Event(self.config.warmup_time, EventType.WARMUP_END))
            
        while self.event_list:
            event = self.get_next_event()
            if event.time > self.config.simulation_time: break
            
            self.update_statistics(event.time)
            self.state.clock = event.time
            
            if event.event_type == EventType.ARRIVAL: self.handle_arrival(event)
            elif event.event_type == EventType.DEPARTURE: self.handle_departure(event)
            elif event.event_type == EventType.WARMUP_END: self.handle_warmup_end()
        
        self.update_statistics(self.config.simulation_time)
        return self.compute_statistics()

    def handle_arrival(self, event: Event):
        queue_before = self.state.queue_length()
        if not self.state.in_warmup:
            self.state.total_arrivals += 1
        packet = Packet(packet_id=event.packet_id, arrival_time=event.time)
        
        if self.config.queue_capacity > 0 and \
           self.state.queue_length() >= self.config.queue_capacity and \
           self.state.all_servers_busy():
            if not self.state.in_warmup: self.state.total_drops += 1
            self.log_event("DROP", event.packet_id, -1, queue_before, self.state.queue_length())
        else:
            idle_server = self.state.get_idle_server()
            if idle_server:
                idle_server.busy = True
                idle_server.current_packet = packet
                idle_server.last_busy_start = event.time
                packet.service_start_time = event.time
                packet.server_id = idle_server.server_id
                svc_time = self.generate_service_time()
                packet.service_time = svc_time
                self.schedule_event(Event(event.time + svc_time, EventType.DEPARTURE, packet.packet_id, idle_server.server_id))
                self.log_event("ARRIVAL (SERVE)", event.packet_id, idle_server.server_id, queue_before, self.state.queue_length())
            else:
                self.state.queue.append(packet)
                self.log_event("ARRIVAL (QUEUE)", event.packet_id, -1, queue_before, self.state.queue_length())
        
        inter_val = self.generate_interarrival_time()
        next_time = event.time + inter_val
        if next_time <= self.config.simulation_time:
            self.packet_counter += 1
            self.schedule_event(Event(next_time, EventType.ARRIVAL, self.packet_counter))

    def handle_departure(self, event: Event):
        queue_before = self.state.queue_length()
        server = self.state.servers[event.server_id]
        
        if server.current_packet:
            if not self.state.in_warmup:
                server.current_packet.departure_time = event.time
                self.state.completed_packets.append(server.current_packet)
                self.state.total_departures += 1
                server.packets_served += 1
                server.total_busy_time += (event.time - server.last_busy_start)
        
        if self.state.queue:
            next_packet = self.state.queue.pop(0)
            server.current_packet = next_packet
            next_packet.service_start_time = event.time
            next_packet.server_id = server.server_id
            server.last_busy_start = event.time
            svc_time = self.generate_service_time()
            next_packet.service_time = svc_time
            self.schedule_event(Event(event.time + svc_time, EventType.DEPARTURE, next_packet.packet_id, server.server_id))
            self.log_event("DEPARTURE (NEXT)", event.packet_id, server.server_id, queue_before, self.state.queue_length())
        else:
            server.busy = False
            server.current_packet = None
            self.log_event("DEPARTURE (IDLE)", event.packet_id, server.server_id, queue_before, self.state.queue_length())

    def generate_interarrival_time(self) -> float:
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
        effective_time = self.config.simulation_time - self.config.warmup_time
        if effective_time <= 0: effective_time = 1.0
        
        waiting_times = [p.waiting_time for p in self.state.completed_packets if p.waiting_time is not None]
        system_times = [p.system_time for p in self.state.completed_packets if p.system_time is not None]
        service_times = [p.service_time for p in self.state.completed_packets]
        
        for server in self.state.servers:
            if server.busy:
                start = max(server.last_busy_start, self.config.warmup_time)
                server.total_busy_time += (self.config.simulation_time - start)

        total_busy = sum(s.total_busy_time for s in self.state.servers)
        utilization = total_busy / (self.config.num_servers * effective_time) if self.config.num_servers > 0 else 0
        
        return {
            'total_arrivals': self.state.total_arrivals,
            'total_departures': self.state.total_departures,
            'total_drops': self.state.total_drops,
            'packets_remaining': self.state.packets_in_system(),
            'num_servers': self.config.num_servers,
            'average_queue_length': self.state.area_under_queue / effective_time,
            'average_system_length': self.state.area_under_system / effective_time,
            'average_waiting_time': np.mean(waiting_times) if waiting_times else 0,
            'average_system_time': np.mean(system_times) if system_times else 0,
            'average_service_time': np.mean(service_times) if service_times else 0,
            'std_waiting_time': np.std(waiting_times) if waiting_times else 0,
            'std_system_time': np.std(system_times) if system_times else 0,
            'max_waiting_time': max(waiting_times) if waiting_times else 0,
            'max_system_time': max(system_times) if system_times else 0,
            'throughput': self.state.total_departures / effective_time,
            'drop_rate': self.state.total_drops / self.state.total_arrivals if self.state.total_arrivals > 0 else 0,
            'server_utilization': utilization,
            'traffic_intensity': self.config.get_traffic_intensity(),
            'waiting_times': waiting_times,
            'system_times': system_times
        }

    # --- Data Export Helpers ---
    def get_event_log_dataframe(self) -> pd.DataFrame:
        if not self.event_log: return pd.DataFrame()
        return pd.DataFrame([vars(e) for e in self.event_log])
    
    def get_packet_table_dataframe(self) -> pd.DataFrame:
        if not self.state.completed_packets: return pd.DataFrame()
        data = []
        for p in self.state.completed_packets:
            d = vars(p).copy()
            d['waiting_time'] = p.waiting_time
            d['system_time'] = p.system_time
            data.append(d)
        return pd.DataFrame(data)
    
    def get_time_series_dataframe(self) -> pd.DataFrame:
        if not self.time_series: return pd.DataFrame()
        return pd.DataFrame(self.time_series)
    
    def get_server_stats_dataframe(self) -> pd.DataFrame:
        total_time = self.config.simulation_time - self.config.warmup_time
        data = []
        for s in self.state.servers:
            data.append({
                'Server ID': s.server_id,
                'Packets Served': s.packets_served,
                'Total Busy Time': round(s.total_busy_time, 4),
                'Utilization': round(s.total_busy_time / total_time, 4) if total_time > 0 else 0
            })
        return pd.DataFrame(data)

    def export_to_json(self) -> str:
        stats = self.compute_statistics()
        stats.pop('waiting_times', None)
        stats.pop('system_times', None)
        export_data = {
            'config': self.config.to_dict(),
            'statistics': stats,
            'event_log': [vars(e) for e in self.event_log],
            'server_stats': self.get_server_stats_dataframe().to_dict(orient='records')
        }
        return json.dumps(export_data, indent=2, default=str)

def compute_mmc_theoretical(lam: float, mu: float, c: int) -> Dict:
    rho = lam / (c * mu)
    if rho >= 1:
        return {'stable': False, 'rho': rho, 'Lq': 0, 'Wq': 0, 'L': 0, 'W': 0}
    sum_terms = sum(((lam/mu)**n) / math.factorial(n) for n in range(c))
    last_term = ((lam/mu)**c) / (math.factorial(c) * (1 - rho))
    P0 = 1 / (sum_terms + last_term)
    Lq = (P0 * ((lam/mu)**c) * rho) / (math.factorial(c) * ((1 - rho)**2))
    L = Lq + (lam/mu)
    Wq = Lq / lam
    W = Wq + (1/mu)
    return {'stable': True, 'rho': rho, 'Lq': Lq, 'Wq': Wq, 'L': L, 'W': W}

def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    if len(data) < 2: return 0.0, 0.0, 0.0
    n = len(data)
    m = np.mean(data)
    se = scipy_stats.sem(data)
    h = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h

def run_replications(config: SimulationConfig, num_reps: int = 10, confidence: float = 0.95) -> Dict:
    results = {'Wq': [], 'Lq': [], 'utilization': [], 'throughput': [], 'drop_rate': [], 'L': [], 'W': []}
    base_seed = config.random_seed if config.random_seed else 12345
    for i in range(num_reps):
        config.random_seed = base_seed + i * 997 
        sim = NetworkSimulator(config)
        res = sim.run()
        results['Wq'].append(res['average_waiting_time'])
        results['Lq'].append(res['average_queue_length'])
        results['utilization'].append(res['server_utilization'])
        results['throughput'].append(res['throughput'])
        results['drop_rate'].append(res['drop_rate'])
        results['L'].append(res['average_system_length'])
        results['W'].append(res['average_system_time'])
    ci_results = {}
    for key, vals in results.items():
        m, lo, hi = compute_confidence_interval(vals, confidence)
        ci_results[key] = {'mean': m, 'lower': lo, 'upper': hi, 'std': np.std(vals), 'values': vals}
    config.random_seed = base_seed
    return ci_results

def run_comparative_analysis(configs: List[SimulationConfig]) -> List[Dict]:
    results = []
    for cfg in configs:
        sim = NetworkSimulator(cfg)
        res = sim.run()
        res.update(cfg.to_dict())
        # Attach time series for plotting in Comparative Mode
        res['time_series_df'] = sim.get_time_series_dataframe()
        results.append(res)
    return results

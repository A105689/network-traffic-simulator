"""
Network Traffic Simulation Engine using Event-Scheduling Approach
Discrete Event Simulation for Probability Course
Supports M/M/1 and M/M/c queuing systems with multiple distributions
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
from io import BytesIO


class EventType(Enum):
    """Types of events in the network simulation"""
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"


class DistributionType(Enum):
    """Supported probability distributions"""
    EXPONENTIAL = "Exponential"
    NORMAL = "Normal"
    UNIFORM = "Uniform"
    WEIBULL = "Weibull"
    GAMMA = "Gamma"
    LOGNORMAL = "Log-normal"
    POISSON = "Poisson"


@dataclass(order=True)
class Event:
    """Event class for the priority queue"""
    time: float
    event_type: EventType = field(compare=False)
    packet_id: int = field(compare=False)
    server_id: int = field(compare=False, default=0)
    
    def __repr__(self):
        return f"Event({self.time:.4f}, {self.event_type.value}, Packet-{self.packet_id}, Server-{self.server_id})"


@dataclass
class Packet:
    """Represents a network packet"""
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
    """Represents a single server in M/M/c system"""
    server_id: int
    busy: bool = False
    current_packet: Optional[Packet] = None
    total_busy_time: float = 0.0
    last_busy_start: float = 0.0
    packets_served: int = 0


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation - all coefficients are dynamic"""
    # Arrival distribution parameters
    arrival_distribution: DistributionType = DistributionType.EXPONENTIAL
    arrival_rate: float = 5.0
    arrival_mean: float = 0.2
    arrival_std: float = 0.05
    arrival_min: float = 0.1
    arrival_max: float = 0.3
    arrival_shape: float = 2.0  # For Weibull/Gamma
    arrival_scale: float = 0.2  # For Weibull/Gamma/Lognormal
    
    # Service distribution parameters
    service_distribution: DistributionType = DistributionType.EXPONENTIAL
    service_rate: float = 8.0
    service_mean: float = 0.125
    service_std: float = 0.03
    service_min: float = 0.05
    service_max: float = 0.2
    service_shape: float = 2.0  # For Weibull/Gamma
    service_scale: float = 0.125  # For Weibull/Gamma/Lognormal
    
    # System parameters
    num_servers: int = 1  # c in M/M/c
    queue_capacity: int = 100
    simulation_time: float = 100.0
    random_seed: Optional[int] = 42
    
    def get_traffic_intensity(self) -> float:
        """Calculate traffic intensity (rho = lambda/(c*mu))"""
        if self.arrival_distribution == DistributionType.EXPONENTIAL:
            effective_arrival_rate = self.arrival_rate
        else:
            effective_arrival_rate = 1 / self.arrival_mean if self.arrival_mean > 0 else 1
            
        if self.service_distribution == DistributionType.EXPONENTIAL:
            effective_service_rate = self.service_rate
        else:
            effective_service_rate = 1 / self.service_mean if self.service_mean > 0 else 1
            
        return effective_arrival_rate / (self.num_servers * effective_service_rate)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            'arrival_distribution': self.arrival_distribution.value,
            'arrival_rate': self.arrival_rate,
            'arrival_mean': self.arrival_mean,
            'arrival_std': self.arrival_std,
            'arrival_min': self.arrival_min,
            'arrival_max': self.arrival_max,
            'arrival_shape': self.arrival_shape,
            'arrival_scale': self.arrival_scale,
            'service_distribution': self.service_distribution.value,
            'service_rate': self.service_rate,
            'service_mean': self.service_mean,
            'service_std': self.service_std,
            'service_min': self.service_min,
            'service_max': self.service_max,
            'service_shape': self.service_shape,
            'service_scale': self.service_scale,
            'num_servers': self.num_servers,
            'queue_capacity': self.queue_capacity,
            'simulation_time': self.simulation_time,
            'random_seed': self.random_seed
        }


class RandomGenerator:
    """Generates random variates from different distributions"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def generate(self, dist_type: DistributionType, **params) -> float:
        """Generate a random variate from the specified distribution"""
        if dist_type == DistributionType.EXPONENTIAL:
            rate = params.get('rate', 1.0)
            return self.rng.exponential(1.0 / rate)
        
        elif dist_type == DistributionType.NORMAL:
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            value = self.rng.normal(mean, std)
            return max(0.001, value)
        
        elif dist_type == DistributionType.UNIFORM:
            low = params.get('min', 0.0)
            high = params.get('max', 1.0)
            return self.rng.uniform(low, high)
        
        elif dist_type == DistributionType.WEIBULL:
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            return scale * self.rng.weibull(shape)
        
        elif dist_type == DistributionType.GAMMA:
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            return self.rng.gamma(shape, scale)
        
        elif dist_type == DistributionType.LOGNORMAL:
            mean = params.get('mean', 0.0)
            sigma = params.get('std', 1.0)
            return self.rng.lognormal(mean, sigma)

        elif dist_type == DistributionType.POISSON:
            lam = params.get('lam', 1.0)  # Changed from 'lambda' to 'lam'
            # Poisson returns integers; ensure we don't return 0.0 for time durations
            val = float(self.rng.poisson(lam))
            return max(0.001, val)
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")


@dataclass
class EventLogEntry:
    """Single entry in the event log table"""
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
    """Current state of the simulation system for M/M/c"""
    clock: float = 0.0
    servers: List[Server] = field(default_factory=list)
    queue: List[Packet] = field(default_factory=list)
    
    # Statistics accumulators
    total_arrivals: int = 0
    total_departures: int = 0
    total_drops: int = 0
    
    # Time-weighted statistics
    area_under_queue: float = 0.0
    area_under_system: float = 0.0
    last_event_time: float = 0.0
    
    # Packet tracking
    completed_packets: List[Packet] = field(default_factory=list)
    
    def queue_length(self) -> int:
        return len(self.queue)
    
    def busy_servers(self) -> int:
        return sum(1 for s in self.servers if s.busy)
    
    def packets_in_system(self) -> int:
        return len(self.queue) + self.busy_servers()
    
    def get_idle_server(self) -> Optional[Server]:
        """Return first idle server, or None if all busy"""
        for server in self.servers:
            if not server.busy:
                return server
        return None
    
    def all_servers_busy(self) -> bool:
        return all(s.busy for s in self.servers)


class NetworkSimulator:
    """
    Discrete Event Simulation of Network Traffic
    Uses Event-Scheduling Approach with Future Event List (FEL)
    Supports M/M/1 and M/M/c queuing systems
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = RandomGenerator(config.random_seed)
        self.state = SystemState()
        self.event_list: List[Event] = []
        self.event_log: List[EventLogEntry] = []
        self.packet_counter = 0
        self.event_counter = 0
        self.time_series: List[Dict] = []
        
    def generate_interarrival_time(self) -> float:
        """Generate inter-arrival time based on configured distribution"""
        config = self.config
        if config.arrival_distribution == DistributionType.EXPONENTIAL:
            return self.rng.generate(DistributionType.EXPONENTIAL, rate=config.arrival_rate)
        elif config.arrival_distribution == DistributionType.NORMAL:
            return self.rng.generate(DistributionType.NORMAL, 
                                     mean=config.arrival_mean, std=config.arrival_std)
        elif config.arrival_distribution == DistributionType.UNIFORM:
            return self.rng.generate(DistributionType.UNIFORM,
                                     min=config.arrival_min, max=config.arrival_max)
        elif config.arrival_distribution == DistributionType.WEIBULL:
            return self.rng.generate(DistributionType.WEIBULL,
                                     shape=config.arrival_shape, scale=config.arrival_scale)
        elif config.arrival_distribution == DistributionType.GAMMA:
            return self.rng.generate(DistributionType.GAMMA,
                                     shape=config.arrival_shape, scale=config.arrival_scale)
        elif config.arrival_distribution == DistributionType.LOGNORMAL:
            return self.rng.generate(DistributionType.LOGNORMAL,
                                     mean=config.arrival_mean, std=config.arrival_std)
        elif config.arrival_distribution == DistributionType.POISSON:
            return self.rng.generate(DistributionType.POISSON, lam=config.arrival_mean) # Changed lambda= to lam=
        else:
            return self.rng.generate(DistributionType.EXPONENTIAL, rate=config.arrival_rate)
    
    def generate_service_time(self) -> float:
        """Generate service time based on configured distribution"""
        config = self.config
        if config.service_distribution == DistributionType.EXPONENTIAL:
            return self.rng.generate(DistributionType.EXPONENTIAL, rate=config.service_rate)
        elif config.service_distribution == DistributionType.NORMAL:
            return self.rng.generate(DistributionType.NORMAL,
                                     mean=config.service_mean, std=config.service_std)
        elif config.service_distribution == DistributionType.UNIFORM:
            return self.rng.generate(DistributionType.UNIFORM,
                                     min=config.service_min, max=config.service_max)
        elif config.service_distribution == DistributionType.WEIBULL:
            return self.rng.generate(DistributionType.WEIBULL,
                                     shape=config.service_shape, scale=config.service_scale)
        elif config.service_distribution == DistributionType.GAMMA:
            return self.rng.generate(DistributionType.GAMMA,
                                     shape=config.service_shape, scale=config.service_scale)
        elif config.service_distribution == DistributionType.LOGNORMAL:
            return self.rng.generate(DistributionType.LOGNORMAL,
                                     mean=config.service_mean, std=config.service_std)
        elif config.service_distribution == DistributionType.POISSON:
            return self.rng.generate(DistributionType.POISSON, lam=config.service_mean) # Changed lambda= to lam=
        else:
            return self.rng.generate(DistributionType.EXPONENTIAL, rate=config.service_rate)
    
    def schedule_event(self, event: Event):
        """Add event to the Future Event List (priority queue)"""
        heapq.heappush(self.event_list, event)
    
    def get_next_event(self) -> Optional[Event]:
        """Get and remove the next event from FEL"""
        if self.event_list:
            return heapq.heappop(self.event_list)
        return None
    
    def update_statistics(self, new_time: float):
        """Update time-weighted statistics"""
        time_delta = new_time - self.state.last_event_time
        self.state.area_under_queue += self.state.queue_length() * time_delta
        self.state.area_under_system += self.state.packets_in_system() * time_delta
        self.state.last_event_time = new_time
    
    def log_event(self, event_type: str, packet_id: int, server_id: int,
                  queue_before: int, queue_after: int):
        """Record event in the event log table"""
        self.event_counter += 1
        entry = EventLogEntry(
            event_number=self.event_counter,
            clock_time=self.state.clock,
            event_type=event_type,
            packet_id=packet_id,
            server_id=server_id,
            queue_length_before=queue_before,
            queue_length_after=queue_after,
            servers_busy=self.state.busy_servers(),
            total_servers=self.config.num_servers,
            packets_in_system=self.state.packets_in_system(),
            cumulative_arrivals=self.state.total_arrivals,
            cumulative_departures=self.state.total_departures,
            cumulative_drops=self.state.total_drops
        )
        self.event_log.append(entry)
        
        self.time_series.append({
            'time': self.state.clock,
            'queue_length': queue_after,
            'packets_in_system': self.state.packets_in_system(),
            'servers_busy': self.state.busy_servers(),
            'server_utilization': self.state.busy_servers() / self.config.num_servers
        })
    
    def handle_arrival(self, event: Event):
        """Process an arrival event"""
        queue_before = self.state.queue_length()
        self.state.total_arrivals += 1
        
        packet = Packet(
            packet_id=event.packet_id,
            arrival_time=event.time
        )
        
        # Check queue capacity (if all servers busy)
        if self.config.queue_capacity > 0 and \
           self.state.queue_length() >= self.config.queue_capacity and \
           self.state.all_servers_busy():
            self.state.total_drops += 1
            self.log_event("ARRIVAL (DROPPED)", event.packet_id, -1,
                          queue_before, self.state.queue_length())
        else:
            idle_server = self.state.get_idle_server()
            if idle_server:
                idle_server.busy = True
                idle_server.current_packet = packet
                idle_server.last_busy_start = event.time
                packet.service_start_time = event.time
                packet.service_time = self.generate_service_time()
                packet.server_id = idle_server.server_id
                
                departure_time = event.time + packet.service_time
                departure_event = Event(departure_time, EventType.DEPARTURE, 
                                        packet.packet_id, idle_server.server_id)
                self.schedule_event(departure_event)
            else:
                self.state.queue.append(packet)
            
            self.log_event("ARRIVAL", event.packet_id, 
                          idle_server.server_id if idle_server else -1,
                          queue_before, self.state.queue_length())
        
        next_arrival_time = event.time + self.generate_interarrival_time()
        if next_arrival_time <= self.config.simulation_time:
            self.packet_counter += 1
            next_arrival = Event(next_arrival_time, EventType.ARRIVAL, self.packet_counter)
            self.schedule_event(next_arrival)
    
    def handle_departure(self, event: Event):
        """Process a departure event"""
        queue_before = self.state.queue_length()
        self.state.total_departures += 1
        
        server = self.state.servers[event.server_id]
        
        if server.current_packet:
            server.current_packet.departure_time = event.time
            self.state.completed_packets.append(server.current_packet)
            server.packets_served += 1
            server.total_busy_time += event.time - server.last_busy_start
        
        if self.state.queue:
            next_packet = self.state.queue.pop(0)
            next_packet.service_start_time = event.time
            next_packet.service_time = self.generate_service_time()
            next_packet.server_id = server.server_id
            server.current_packet = next_packet
            server.last_busy_start = event.time
            
            departure_time = event.time + next_packet.service_time
            departure_event = Event(departure_time, EventType.DEPARTURE, 
                                    next_packet.packet_id, server.server_id)
            self.schedule_event(departure_event)
        else:
            server.busy = False
            server.current_packet = None
        
        self.log_event("DEPARTURE", event.packet_id, event.server_id,
                      queue_before, self.state.queue_length())
    
    def initialize(self):
        """Initialize the simulation"""
        self.state = SystemState()
        self.state.servers = [Server(server_id=i) for i in range(self.config.num_servers)]
        self.event_list = []
        self.event_log = []
        self.time_series = []
        self.packet_counter = 0
        self.event_counter = 0
        
        self.rng = RandomGenerator(self.config.random_seed)
        
        self.packet_counter += 1
        first_arrival = Event(0.0, EventType.ARRIVAL, self.packet_counter)
        self.schedule_event(first_arrival)
    
    def run(self) -> Dict:
        """Execute the simulation"""
        self.initialize()
        
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
        
        self.update_statistics(self.config.simulation_time)
        
        return self.compute_statistics()
    
    def compute_statistics(self) -> Dict:
        """Compute summary statistics from the simulation"""
        total_time = self.config.simulation_time
        
        waiting_times = [p.waiting_time for p in self.state.completed_packets 
                        if p.waiting_time is not None]
        system_times = [p.system_time for p in self.state.completed_packets
                       if p.system_time is not None]
        service_times = [p.service_time for p in self.state.completed_packets]
        
        total_busy_time = sum(s.total_busy_time for s in self.state.servers)
        for server in self.state.servers:
            if server.busy:
                total_busy_time += total_time - server.last_busy_start
        
        server_utilization = total_busy_time / (self.config.num_servers * total_time) if total_time > 0 else 0
        
        per_server_stats = []
        for server in self.state.servers:
            busy_time = server.total_busy_time
            if server.busy:
                busy_time += total_time - server.last_busy_start
            per_server_stats.append({
                'server_id': server.server_id,
                'packets_served': server.packets_served,
                'utilization': busy_time / total_time if total_time > 0 else 0
            })
        
        stats = {
            'total_arrivals': self.state.total_arrivals,
            'total_departures': self.state.total_departures,
            'total_drops': self.state.total_drops,
            'packets_remaining': self.state.packets_in_system(),
            'num_servers': self.config.num_servers,
            
            'average_queue_length': self.state.area_under_queue / total_time if total_time > 0 else 0,
            'average_system_length': self.state.area_under_system / total_time if total_time > 0 else 0,
            
            'average_waiting_time': np.mean(waiting_times) if waiting_times else 0,
            'average_system_time': np.mean(system_times) if system_times else 0,
            'average_service_time': np.mean(service_times) if service_times else 0,
            
            'std_waiting_time': np.std(waiting_times) if waiting_times else 0,
            'std_system_time': np.std(system_times) if system_times else 0,
            
            'max_waiting_time': max(waiting_times) if waiting_times else 0,
            'max_system_time': max(system_times) if system_times else 0,
            
            'throughput': self.state.total_departures / total_time if total_time > 0 else 0,
            'drop_rate': self.state.total_drops / self.state.total_arrivals if self.state.total_arrivals > 0 else 0,
            'server_utilization': server_utilization,
            
            'traffic_intensity': self.config.get_traffic_intensity(),
            'simulation_time': total_time,
            'queue_capacity': self.config.queue_capacity,
            
            'per_server_stats': per_server_stats,
            'waiting_times': waiting_times,
            'system_times': system_times,
        }
        
        return stats
    
    def get_event_log_dataframe(self) -> pd.DataFrame:
        """Convert event log to pandas DataFrame"""
        if not self.event_log:
            return pd.DataFrame()
        
        data = []
        for entry in self.event_log:
            data.append({
                'Event #': entry.event_number,
                'Clock Time': round(entry.clock_time, 4),
                'Event Type': entry.event_type,
                'Packet ID': entry.packet_id,
                'Server ID': entry.server_id if entry.server_id >= 0 else 'N/A',
                'Queue (Before)': entry.queue_length_before,
                'Queue (After)': entry.queue_length_after,
                'Servers Busy': f"{entry.servers_busy}/{entry.total_servers}",
                'In System': entry.packets_in_system,
                'Total Arrivals': entry.cumulative_arrivals,
                'Total Departures': entry.cumulative_departures,
                'Total Drops': entry.cumulative_drops
            })
        
        return pd.DataFrame(data)
    
    def get_packet_table_dataframe(self) -> pd.DataFrame:
        """Get detailed packet-level statistics"""
        if not self.state.completed_packets:
            return pd.DataFrame()
        
        data = []
        for packet in self.state.completed_packets:
            data.append({
                'Packet ID': packet.packet_id,
                'Server ID': packet.server_id,
                'Arrival Time': round(packet.arrival_time, 4),
                'Service Start': round(packet.service_start_time, 4) if packet.service_start_time else 'N/A',
                'Departure Time': round(packet.departure_time, 4) if packet.departure_time else 'N/A',
                'Waiting Time': round(packet.waiting_time, 4) if packet.waiting_time else 'N/A',
                'Service Time': round(packet.service_time, 4),
                'System Time': round(packet.system_time, 4) if packet.system_time else 'N/A'
            })
        
        return pd.DataFrame(data)
    
    def get_time_series_dataframe(self) -> pd.DataFrame:
        """Get time series data for visualization"""
        if not self.time_series:
            return pd.DataFrame()
        return pd.DataFrame(self.time_series)
    
    def get_server_stats_dataframe(self) -> pd.DataFrame:
        """Get per-server statistics"""
        total_time = self.config.simulation_time
        data = []
        for server in self.state.servers:
            busy_time = server.total_busy_time
            if server.busy:
                busy_time += total_time - server.last_busy_start
            data.append({
                'Server ID': server.server_id,
                'Packets Served': server.packets_served,
                'Total Busy Time': round(busy_time, 4),
                'Utilization': round(busy_time / total_time, 4) if total_time > 0 else 0
            })
        return pd.DataFrame(data)
    
    def export_to_json(self) -> str:
        """Export simulation results to JSON for replay"""
        export_data = {
            'config': self.config.to_dict(),
            'statistics': {k: v for k, v in self.compute_statistics().items() 
                          if k not in ['waiting_times', 'system_times', 'per_server_stats']},
            'event_log': self.get_event_log_dataframe().to_dict(orient='records'),
            'packet_table': self.get_packet_table_dataframe().to_dict(orient='records'),
            'time_series': self.time_series
        }
        return json.dumps(export_data, indent=2)


def compute_mmc_theoretical(lam: float, mu: float, c: int) -> Dict:
    """
    Compute theoretical M/M/c queue metrics
    lam: arrival rate
    mu: service rate per server
    c: number of servers
    """
    rho = lam / (c * mu)
    
    if rho >= 1:
        return {
            'stable': False,
            'rho': rho,
            'Lq': float('inf'),
            'L': float('inf'),
            'Wq': float('inf'),
            'W': float('inf'),
            'P0': 0
        }
    
    a = lam / mu
    
    sum_terms = sum((a ** n) / math.factorial(n) for n in range(c))
    last_term = (a ** c) / (math.factorial(c) * (1 - rho))
    P0 = 1 / (sum_terms + last_term)
    
    Lq = (P0 * (a ** c) * rho) / (math.factorial(c) * ((1 - rho) ** 2))
    L = Lq + a
    Wq = Lq / lam
    W = Wq + 1/mu
    
    return {
        'stable': True,
        'rho': rho,
        'Lq': Lq,
        'L': L,
        'Wq': Wq,
        'W': W,
        'P0': P0
    }


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute confidence interval for the mean of data
    Returns: (mean, lower_bound, upper_bound)
    """
    if not data or len(data) < 2:
        return (0, 0, 0)
    
    n = len(data)
    mean = np.mean(data)
    se = scipy_stats.sem(data)
    
    h = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return (mean, mean - h, mean + h)


def run_replications(config: SimulationConfig, num_replications: int = 10, 
                     confidence_level: float = 0.95) -> Dict:
    """
    Run multiple simulation replications and compute confidence intervals
    """
    results = {
        'Lq': [], 'L': [], 'Wq': [], 'W': [], 
        'utilization': [], 'throughput': [], 'drop_rate': []
    }
    
    base_seed = config.random_seed if config.random_seed else 42
    
    for i in range(num_replications):
        config.random_seed = base_seed + i
        sim = NetworkSimulator(config)
        stats = sim.run()
        
        results['Lq'].append(stats['average_queue_length'])
        results['L'].append(stats['average_system_length'])
        results['Wq'].append(stats['average_waiting_time'])
        results['W'].append(stats['average_system_time'])
        results['utilization'].append(stats['server_utilization'])
        results['throughput'].append(stats['throughput'])
        results['drop_rate'].append(stats['drop_rate'])
    
    ci_results = {}
    for metric, values in results.items():
        mean, lower, upper = compute_confidence_interval(values, confidence_level)
        ci_results[metric] = {
            'mean': mean,
            'std': np.std(values),
            'lower': lower,
            'upper': upper,
            'values': values
        }
    
    config.random_seed = base_seed
    
    return ci_results


def run_comparative_analysis(configs: List[SimulationConfig]) -> List[Dict]:
    """
    Run multiple simulations with different configurations for comparison
    """
    results = []
    for i, config in enumerate(configs):
        sim = NetworkSimulator(config)
        stats = sim.run()
        stats['config_id'] = i
        stats['config'] = config.to_dict()
        results.append(stats)
    return results

# ─────────────────────────────────────────────────────────────────────────
# Ported from SENTINEL: sentinel/agents/anomaly/kitsune.py  (unchanged)
# Packet-level / timing feature source for SIH PS 26153 (IAT, jitter, per-5-tuple).
# ─────────────────────────────────────────────────────────────────────────
"""
SENTINEL Agent 1 — Kitsune Feature Extractor

Implements the 115 incremental statistics used by the Kitsune framework
to profile network traffic at various time scales.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class IncrementalStat:
    """
    A single damped incremental statistic (weight, mean, variance).
    Based on the Kitsune/AfterImage logic.
    """
    def __init__(self, Lambda: float):
        self.Lambda = Lambda
        self.last_timestamp = -1.0
        self.weight = 0.0
        self.mean = 0.0
        self.variance = 0.0

    def update(self, value: float, timestamp: float):
        if self.last_timestamp == -1.0:
            self.weight = 1.0
            self.mean = value
            self.variance = 0.0
            self.last_timestamp = timestamp
            return

        # Damping factor
        delta_t = timestamp - self.last_timestamp
        factor = math.pow(2, -self.Lambda * delta_t)

        # Update damped statistics
        self.weight = self.weight * factor + 1.0
        
        old_mean = self.mean
        self.mean = (1.0 - (1.0 / self.weight)) * self.mean + (1.0 / self.weight) * value
        
        # Incremental variance (Welford's variation with damping)
        self.variance = self.variance * factor + (value - old_mean) * (value - self.mean)
        if self.variance < 0:
            self.variance = 0.0

        self.last_timestamp = timestamp

    def get_stats(self) -> List[float]:
        return [self.weight, self.mean, self.variance]

class IncrementalStat2D:
    """
    Two-dimensional damped incremental statistics for tracking relationships.
    (weight, mean_x, mean_y, var_x, var_y, covariance).
    """
    def __init__(self, Lambda: float):
        self.Lambda = Lambda
        self.last_timestamp = -1.0
        self.weight = 0.0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.var_x = 0.0
        self.var_y = 0.0
        self.cov = 0.0

    def update(self, val_x: float, val_y: float, timestamp: float):
        if self.last_timestamp == -1.0:
            self.weight = 1.0
            self.mean_x = val_x
            self.mean_y = val_y
            self.var_x = 0.0
            self.var_y = 0.0
            self.cov = 0.0
            self.last_timestamp = timestamp
            return

        delta_t = timestamp - self.last_timestamp
        factor = math.pow(2, -self.Lambda * delta_t)

        self.weight = self.weight * factor + 1.0
        
        old_mean_x = self.mean_x
        old_mean_y = self.mean_y
        
        self.mean_x = (1.0 - (1.0 / self.weight)) * self.mean_x + (1.0 / self.weight) * val_x
        self.mean_y = (1.0 - (1.0 / self.weight)) * self.mean_y + (1.0 / self.weight) * val_y
        
        self.var_x = self.var_x * factor + (val_x - old_mean_x) * (val_x - self.mean_x)
        self.var_y = self.var_y * factor + (val_y - old_mean_y) * (val_y - self.mean_y)
        self.cov = self.cov * factor + (val_x - old_mean_x) * (val_y - self.mean_y)
        
        if self.var_x < 0: self.var_x = 0.0
        if self.var_y < 0: self.var_y = 0.0

        self.last_timestamp = timestamp

    def get_stats(self) -> List[float]:
        # std_x, std_y
        std_x = math.sqrt(self.var_x)
        std_y = math.sqrt(self.var_y)
        
        # magnitude
        mag = math.sqrt(self.mean_x**2 + self.mean_y**2)
        
        # radius
        radius = math.sqrt(self.var_x + self.var_y)
        
        # PCC (Pearson Correlation Coefficient)
        pcc = 0.0
        if std_x > 0 and std_y > 0:
            pcc = self.cov / (std_x * std_y)
            # Clamp PCC
            pcc = max(min(pcc, 1.0), -1.0)
            
        return [self.weight, self.mean_x, std_x, mag, radius, self.cov, pcc]

class KitsuneExtractor:
    """
    Main extractor that maintains a set of incremental statistics
    to produce the 115-feature vector.
    """
    def __init__(self):
        # Time windows (lambdas)
        self.lambdas = [5.0, 3.0, 1.0, 0.1, 0.01]
        
        # Stats storage
        self.MI = {}    # src_ip -> [IncrementalStat x 5]
        self.H = {}     # src_mac -> [IncrementalStat x 5]
        self.HH = {}    # (src_ip, dst_ip) -> [IncrementalStat2D x 5]
        self.HH_jit = {} # (src_ip, dst_ip) -> [IncrementalStat x 5]
        self.HpHp = {}  # (src_ip, src_port, dst_ip, dst_port) -> [IncrementalStat2D x 5]
        
        # To track jitter
        self.last_arrival_times = {} # (src_ip, dst_ip) -> timestamp

    def extract(self, 
                src_ip: str, src_mac: str, src_port: int,
                dst_ip: str, dst_mac: str, dst_port: int,
                pkt_len: int, timestamp: float) -> np.ndarray:
        
        # 1. MI (Source IP)
        if src_ip not in self.MI:
            self.MI[src_ip] = [IncrementalStat(l) for l in self.lambdas]
        for s in self.MI[src_ip]:
            s.update(pkt_len, timestamp)
            
        # 2. H (Source MAC)
        if src_mac not in self.H:
            self.H[src_mac] = [IncrementalStat(l) for l in self.lambdas]
        for s in self.H[src_mac]:
            s.update(pkt_len, timestamp)
            
        # 3. HH (Source IP -> Dest IP)
        hh_key = (src_ip, dst_ip)
        if hh_key not in self.HH:
            self.HH[hh_key] = [IncrementalStat2D(l) for l in self.lambdas]
        # In Kitsune, HH usually tracks (pkt_len, same_len?) or similar.
        # Most implementations use (pkt_len, pkt_len) for HH and HpHp 
        # to get magnitude and radius of the flow.
        for s in self.HH[hh_key]:
            s.update(pkt_len, pkt_len, timestamp)
            
        # 4. HH_jit (Jitter)
        if hh_key not in self.HH_jit:
            self.HH_jit[hh_key] = [IncrementalStat(l) for l in self.lambdas]
            self.last_arrival_times[hh_key] = timestamp
            jitter = 0.0
        else:
            jitter = abs(timestamp - self.last_arrival_times[hh_key])
            self.last_arrival_times[hh_key] = timestamp
            
        for s in self.HH_jit[hh_key]:
            s.update(jitter, timestamp)
            
        # 5. HpHp (Source IP:Port -> Dest IP:Port)
        hphp_key = (src_ip, src_port, dst_ip, dst_port)
        if hphp_key not in self.HpHp:
            self.HpHp[hphp_key] = [IncrementalStat2D(l) for l in self.lambdas]
        for s in self.HpHp[hphp_key]:
            s.update(pkt_len, pkt_len, timestamp)
            
        # Assemble feature vector (115 features)
        features = []
        
        # MI_dir (15)
        for i in range(5):
            features.extend(self.MI[src_ip][i].get_stats())
            
        # H (15)
        for i in range(5):
            features.extend(self.H[src_mac][i].get_stats())
            
        # HH (35)
        for i in range(5):
            features.extend(self.HH[hh_key][i].get_stats())
            
        # HH_jit (15)
        for i in range(5):
            features.extend(self.HH_jit[hh_key][i].get_stats())
            
        # HpHp (35)
        for i in range(5):
            features.extend(self.HpHp[hphp_key][i].get_stats())
            
        return np.array(features, dtype=np.float32)

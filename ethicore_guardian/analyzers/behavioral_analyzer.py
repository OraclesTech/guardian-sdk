"""
Ethicore Engine™ - Guardian - Behavioral Analyzer (Server-Side)
Detects automation, bots, and unnatural request patterns
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RequestData:
    """Individual request data point"""
    timestamp: float
    text_length: int
    request_size: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    content_hash: Optional[str] = None


@dataclass
class BehavioralProfile:
    """Behavioral profile for a user/session"""
    user_id: str
    session_id: Optional[str] = None
    
    # Request timing
    request_intervals: deque = field(default_factory=lambda: deque(maxlen=50))
    request_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Content patterns  
    content_hashes: deque = field(default_factory=lambda: deque(maxlen=20))
    request_sizes: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Session tracking
    session_start: float = field(default_factory=time.time)
    total_requests: int = 0
    
    # Anomaly indicators
    rapid_fire_count: int = 0
    identical_intervals: int = 0
    large_payload_count: int = 0
    duplicate_content_count: int = 0


@dataclass
class BehavioralAnalysisResult:
    """Result of behavioral analysis"""
    is_suspicious: bool
    anomaly_score: float  # 0-100 scale
    confidence: float  # 0-1 scale
    verdict: str  # ALLOW, CHALLENGE, BLOCK
    behavioral_signals: List[str]
    profile_summary: Dict[str, Any]
    analysis: Dict[str, Any]


class BehavioralAnalyzer:
    """
    Server-side behavioral analysis for API request patterns
    
    Detects:
    - Automation and bot behavior
    - Rapid-fire request patterns  
    - Unnatural timing consistency
    - Bulk content operations
    - Rate limiting abuse
    - Copy-paste attacks (repeated content)
    """
    
    def __init__(self, max_profiles: int = 1000):
        self.initialized = False
        self.profiles: Dict[str, BehavioralProfile] = {}
        self.max_profiles = max_profiles
        
        # Configuration thresholds
        self.config = {
            # Timing thresholds (seconds)
            "rapid_fire_threshold": 0.5,      # <500ms between requests
            "mechanical_consistency": 0.1,    # <100ms variance = bot
            "human_min_variance": 0.3,        # Humans vary by >300ms
            
            # Request frequency
            "max_requests_per_minute": 30,
            "burst_request_count": 5,          # 5+ requests in rapid succession
            
            # Content analysis
            "duplicate_content_threshold": 3,  # 3+ identical requests
            "large_payload_threshold": 5000,   # >5KB requests
            
            # Session patterns
            "session_duration_threshold": 3600,  # 1 hour sessions
            "bot_session_indicators": 10,       # Multiple bot signals
            
            # Scoring weights
            "rapid_fire_weight": 15,
            "consistency_weight": 20,
            "duplicate_weight": 10,
            "burst_weight": 12,
            "large_payload_weight": 8
        }
        
        logger.info("🤖 Behavioral Analyzer initialized (server-side)")
    
    def initialize(self) -> bool:
        """Initialize behavioral analyzer"""
        try:
            # Clean up old profiles periodically
            self._cleanup_old_profiles()
            self.initialized = True
            logger.info("✅ Behavioral Analyzer: Initialization complete")
            return True
        except Exception as e:
            logger.error(f"❌ Behavioral Analyzer initialization failed: {e}")
            return False
    
    def analyze(self, text: str, metadata: Dict[str, Any] = None) -> BehavioralAnalysisResult:
        """
        Analyze request for behavioral anomalies
        
        Args:
            text: Request content/prompt
            metadata: Request metadata (user_id, session_id, etc.)
            
        Returns:
            BehavioralAnalysisResult with anomaly analysis
        """
        if not self.initialized:
            logger.warning("Behavioral analyzer not initialized")
            return self._empty_result()
        
        if not metadata:
            metadata = {}
        
        # Create request data
        request_data = self._extract_request_data(text, metadata)
        
        # Get or create user profile
        profile = self._get_or_create_profile(request_data)
        
        # Update profile with new request
        self._update_profile(profile, request_data)
        
        # Analyze behavioral patterns
        analysis_result = self._analyze_behavioral_patterns(profile, request_data)
        
        return analysis_result
    
    def _extract_request_data(self, text: str, metadata: Dict[str, Any]) -> RequestData:
        """Extract request data from input"""
        current_time = time.time()
        
        # Calculate content hash for duplicate detection
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        return RequestData(
            timestamp=current_time,
            text_length=len(text),
            request_size=len(text.encode('utf-8')),
            user_id=metadata.get('user_id', 'anonymous'),
            session_id=metadata.get('session_id'),
            ip_address=metadata.get('ip_address'),
            user_agent=metadata.get('user_agent'),
            content_hash=content_hash
        )
    
    def _get_or_create_profile(self, request_data: RequestData) -> BehavioralProfile:
        """Get existing profile or create new one"""
        profile_key = f"{request_data.user_id}:{request_data.session_id or 'default'}"
        
        if profile_key not in self.profiles:
            # Create new profile
            self.profiles[profile_key] = BehavioralProfile(
                user_id=request_data.user_id,
                session_id=request_data.session_id
            )
            
            # Cleanup if too many profiles
            if len(self.profiles) > self.max_profiles:
                self._cleanup_old_profiles()
        
        return self.profiles[profile_key]
    
    def _update_profile(self, profile: BehavioralProfile, request_data: RequestData):
        """Update behavioral profile with new request"""
        # Update request timing
        if profile.request_timestamps:
            last_timestamp = profile.request_timestamps[-1]
            interval = request_data.timestamp - last_timestamp
            profile.request_intervals.append(interval)
        
        profile.request_timestamps.append(request_data.timestamp)
        profile.request_sizes.append(request_data.request_size)
        profile.content_hashes.append(request_data.content_hash)
        profile.total_requests += 1
        
        # Check for specific patterns
        self._check_rapid_fire(profile, request_data)
        self._check_content_duplication(profile, request_data)
        self._check_large_payloads(profile, request_data)
    
    def _check_rapid_fire(self, profile: BehavioralProfile, request_data: RequestData):
        """Check for rapid-fire request patterns"""
        if len(profile.request_intervals) >= 2:
            recent_intervals = list(profile.request_intervals)[-5:]  # Last 5 intervals
            
            # Count rapid intervals
            rapid_count = sum(1 for interval in recent_intervals 
                            if interval < self.config["rapid_fire_threshold"])
            
            if rapid_count >= self.config["burst_request_count"] - 1:
                profile.rapid_fire_count += 1
    
    def _check_content_duplication(self, profile: BehavioralProfile, request_data: RequestData):
        """Check for duplicate content patterns"""
        if len(profile.content_hashes) >= 3:
            recent_hashes = list(profile.content_hashes)[-10:]  # Last 10 requests
            hash_counts = {}
            
            for h in recent_hashes:
                hash_counts[h] = hash_counts.get(h, 0) + 1
            
            # Check for duplicates
            max_duplicates = max(hash_counts.values())
            if max_duplicates >= self.config["duplicate_content_threshold"]:
                profile.duplicate_content_count += 1
    
    def _check_large_payloads(self, profile: BehavioralProfile, request_data: RequestData):
        """Check for large payload patterns"""
        if request_data.request_size > self.config["large_payload_threshold"]:
            profile.large_payload_count += 1
    
    def _analyze_behavioral_patterns(self, profile: BehavioralProfile, request_data: RequestData) -> BehavioralAnalysisResult:
        """Analyze behavioral patterns and calculate anomaly score"""
        
        anomaly_score = 0.0
        behavioral_signals = []
        analysis_details = {}
        
        # 1. Timing Pattern Analysis
        timing_analysis = self._analyze_timing_patterns(profile)
        anomaly_score += timing_analysis["score"]
        behavioral_signals.extend(timing_analysis["signals"])
        analysis_details["timing"] = timing_analysis
        
        # 2. Request Frequency Analysis
        frequency_analysis = self._analyze_request_frequency(profile)
        anomaly_score += frequency_analysis["score"]
        behavioral_signals.extend(frequency_analysis["signals"])
        analysis_details["frequency"] = frequency_analysis
        
        # 3. Content Pattern Analysis
        content_analysis = self._analyze_content_patterns(profile)
        anomaly_score += content_analysis["score"]
        behavioral_signals.extend(content_analysis["signals"])
        analysis_details["content"] = content_analysis
        
        # 4. Session Pattern Analysis
        session_analysis = self._analyze_session_patterns(profile)
        anomaly_score += session_analysis["score"]
        behavioral_signals.extend(session_analysis["signals"])
        analysis_details["session"] = session_analysis
        
        # Cap anomaly score at 100
        anomaly_score = min(100.0, anomaly_score)
        
        # Calculate confidence and verdict
        confidence = self._calculate_confidence(profile, anomaly_score)
        verdict = self._determine_verdict(anomaly_score, behavioral_signals)
        
        # Profile summary
        profile_summary = {
            "user_id": profile.user_id,
            "session_id": profile.session_id,
            "total_requests": profile.total_requests,
            "session_duration": time.time() - profile.session_start,
            "avg_request_interval": statistics.mean(profile.request_intervals) if profile.request_intervals else 0,
            "request_frequency": len(profile.request_timestamps) / max(1, (time.time() - profile.session_start) / 60),  # per minute
        }
        
        return BehavioralAnalysisResult(
            is_suspicious=anomaly_score >= 30.0,
            anomaly_score=anomaly_score,
            confidence=confidence,
            verdict=verdict,
            behavioral_signals=behavioral_signals,
            profile_summary=profile_summary,
            analysis=analysis_details
        )
    
    def _analyze_timing_patterns(self, profile: BehavioralProfile) -> Dict[str, Any]:
        """Analyze timing patterns for bot-like behavior"""
        if len(profile.request_intervals) < 3:
            return {"score": 0.0, "signals": []}
        
        intervals = list(profile.request_intervals)
        score = 0.0
        signals = []
        
        # Check for mechanical consistency
        if len(intervals) >= 5:
            variance = statistics.variance(intervals[-5:])  # Last 5 intervals
            
            if variance < self.config["mechanical_consistency"]:
                score += self.config["consistency_weight"]
                signals.append("mechanical_timing_consistency")
            
            # Check for identical intervals (perfect bot behavior)
            recent_intervals = [round(i, 2) for i in intervals[-5:]]
            if len(set(recent_intervals)) <= 2:  # Only 1-2 unique intervals
                score += 15
                signals.append("identical_intervals")
        
        # Check for rapid-fire patterns
        rapid_intervals = sum(1 for i in intervals[-10:] if i < self.config["rapid_fire_threshold"])
        if rapid_intervals >= 3:
            score += self.config["rapid_fire_weight"]
            signals.append("rapid_fire_requests")
        
        # Check for lack of human variance
        if len(intervals) >= 10:
            overall_variance = statistics.variance(intervals)
            if overall_variance < self.config["human_min_variance"]:
                score += 10
                signals.append("insufficient_human_variance")
        
        return {
            "score": score,
            "signals": signals,
            "variance": statistics.variance(intervals) if len(intervals) > 1 else 0,
            "rapid_intervals": rapid_intervals,
            "avg_interval": statistics.mean(intervals)
        }
    
    def _analyze_request_frequency(self, profile: BehavioralProfile) -> Dict[str, Any]:
        """Analyze request frequency patterns"""
        current_time = time.time()
        score = 0.0
        signals = []
        
        # Calculate requests per minute
        session_duration = max(1, current_time - profile.session_start)
        requests_per_minute = profile.total_requests / (session_duration / 60)
        
        if requests_per_minute > self.config["max_requests_per_minute"]:
            score += 20
            signals.append("excessive_request_frequency")
        
        # Check for burst patterns
        if len(profile.request_timestamps) >= 5:
            recent_timestamps = list(profile.request_timestamps)[-5:]
            time_span = recent_timestamps[-1] - recent_timestamps[0]
            
            if time_span < 3.0:  # 5 requests in < 3 seconds
                score += self.config["burst_weight"]
                signals.append("burst_request_pattern")
        
        return {
            "score": score,
            "signals": signals,
            "requests_per_minute": requests_per_minute,
            "total_requests": profile.total_requests,
            "session_duration": session_duration
        }
    
    def _analyze_content_patterns(self, profile: BehavioralProfile) -> Dict[str, Any]:
        """Analyze content patterns for automation"""
        score = 0.0
        signals = []
        
        # Duplicate content detection
        if profile.duplicate_content_count > 0:
            score += self.config["duplicate_weight"] * profile.duplicate_content_count
            signals.append("duplicate_content_detected")
        
        # Large payload patterns
        if profile.large_payload_count > 2:
            score += self.config["large_payload_weight"]
            signals.append("large_payload_pattern")
        
        # Size consistency (bot-like uniform sizes)
        if len(profile.request_sizes) >= 5:
            size_variance = statistics.variance(profile.request_sizes)
            avg_size = statistics.mean(profile.request_sizes)
            
            # If variance is very low relative to average size
            if avg_size > 100 and size_variance < (avg_size * 0.1):
                score += 8
                signals.append("uniform_request_sizes")
        
        return {
            "score": score,
            "signals": signals,
            "duplicate_count": profile.duplicate_content_count,
            "large_payload_count": profile.large_payload_count,
            "avg_request_size": statistics.mean(profile.request_sizes) if profile.request_sizes else 0
        }
    
    def _analyze_session_patterns(self, profile: BehavioralProfile) -> Dict[str, Any]:
        """Analyze session-level patterns"""
        current_time = time.time()
        session_duration = current_time - profile.session_start
        score = 0.0
        signals = []
        
        # Very long sessions without variance
        if session_duration > self.config["session_duration_threshold"]:
            if len(profile.request_intervals) > 10:
                avg_variance = statistics.variance(profile.request_intervals)
                if avg_variance < 0.5:  # Very consistent over long period
                    score += 15
                    signals.append("long_session_low_variance")
        
        # Perfect automation indicators
        automation_indicators = sum([
            profile.rapid_fire_count,
            profile.duplicate_content_count,
            profile.large_payload_count
        ])
        
        if automation_indicators >= self.config["bot_session_indicators"]:
            score += 25
            signals.append("high_automation_indicators")
        
        return {
            "score": score,
            "signals": signals,
            "session_duration": session_duration,
            "automation_indicators": automation_indicators
        }
    
    def _calculate_confidence(self, profile: BehavioralProfile, anomaly_score: float) -> float:
        """Calculate confidence in the behavioral analysis"""
        # Base confidence on amount of data
        data_confidence = min(1.0, profile.total_requests / 10.0)
        
        # Boost confidence for clear patterns
        pattern_confidence = min(1.0, anomaly_score / 50.0)
        
        # Session length adds confidence
        session_confidence = min(1.0, (time.time() - profile.session_start) / 300.0)  # 5 minutes
        
        # Combine confidences
        overall_confidence = (data_confidence + pattern_confidence + session_confidence) / 3.0
        
        return round(min(1.0, overall_confidence), 3)
    
    def _determine_verdict(self, anomaly_score: float, signals: List[str]) -> str:
        """Determine final verdict based on score and signals"""
        # Critical automation signals = immediate block
        critical_signals = {
            "mechanical_timing_consistency",
            "identical_intervals",
            "high_automation_indicators"
        }
        
        if any(signal in critical_signals for signal in signals):
            return "BLOCK"
        
        # Score-based verdicts
        if anomaly_score >= 70:
            return "BLOCK"
        elif anomaly_score >= 40:
            return "CHALLENGE"
        else:
            return "ALLOW"
    
    def _cleanup_old_profiles(self):
        """Clean up old behavioral profiles"""
        current_time = time.time()
        cutoff_time = current_time - 7200  # 2 hours
        
        profiles_to_remove = []
        for key, profile in self.profiles.items():
            if profile.session_start < cutoff_time:
                profiles_to_remove.append(key)
        
        for key in profiles_to_remove:
            del self.profiles[key]
        
        if profiles_to_remove:
            logger.info(f"🧹 Cleaned up {len(profiles_to_remove)} old behavioral profiles")
    
    def _empty_result(self) -> BehavioralAnalysisResult:
        """Return empty result when analysis cannot be performed"""
        return BehavioralAnalysisResult(
            is_suspicious=False,
            anomaly_score=0.0,
            confidence=0.0,
            verdict="ALLOW",
            behavioral_signals=[],
            profile_summary={},
            analysis={}
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get behavioral analyzer status"""
        return {
            "initialized": self.initialized,
            "active_profiles": len(self.profiles),
            "max_profiles": self.max_profiles,
            "config": self.config
        }
    
    def get_profile_summary(self, user_id: str, session_id: str = None) -> Optional[Dict[str, Any]]:
        """Get summary of a specific user profile"""
        profile_key = f"{user_id}:{session_id or 'default'}"
        profile = self.profiles.get(profile_key)
        
        if not profile:
            return None
        
        return {
            "user_id": profile.user_id,
            "session_id": profile.session_id,
            "total_requests": profile.total_requests,
            "session_duration": time.time() - profile.session_start,
            "avg_interval": statistics.mean(profile.request_intervals) if profile.request_intervals else 0,
            "request_frequency": profile.total_requests / max(1, (time.time() - profile.session_start) / 60),
            "anomaly_indicators": {
                "rapid_fire": profile.rapid_fire_count,
                "duplicates": profile.duplicate_content_count,
                "large_payloads": profile.large_payload_count
            }
        }


# CLI testing interface  
if __name__ == "__main__":
    analyzer = BehavioralAnalyzer()
    analyzer.initialize()
    
    # Test cases
    test_cases = [
        ("Hello, how are you?", {"user_id": "user1"}),
        ("Short request", {"user_id": "user2"}),
        ("Automated request", {"user_id": "bot1"}),
    ]
    
    print("\n🤖 Running behavioral analyzer tests...\n")
    
    for text, metadata in test_cases:
        result = analyzer.analyze(text, metadata)
        print(f"Text: {text}")
        print(f"  Anomaly Score: {result.anomaly_score:.1f}")
        print(f"  Verdict: {result.verdict}")
        print(f"  Signals: {', '.join(result.behavioral_signals)}")
        print()
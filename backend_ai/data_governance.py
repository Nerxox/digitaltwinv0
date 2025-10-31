"""
Data governance and quality management for Digital Twin Energy system.
"""
import os
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataQualityRule:
    """Data quality rule definition."""
    name: str
    description: str
    check_function: callable
    severity: str  # 'error', 'warning', 'info'
    enabled: bool = True

@dataclass
class DataQualityResult:
    """Data quality check result."""
    rule_name: str
    passed: bool
    message: str
    severity: str
    timestamp: datetime
    details: Dict[str, Any] = None

class DataQualityChecker:
    """Comprehensive data quality checking system."""
    
    def __init__(self):
        self.rules = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default data quality rules."""
        
        # Completeness rules
        self.add_rule(DataQualityRule(
            name="no_null_timestamps",
            description="No null timestamps allowed",
            check_function=self._check_no_null_timestamps,
            severity="error"
        ))
        
        self.add_rule(DataQualityRule(
            name="no_null_power_values",
            description="No null power values allowed",
            check_function=self._check_no_null_power_values,
            severity="error"
        ))
        
        # Validity rules
        self.add_rule(DataQualityRule(
            name="valid_timestamp_format",
            description="Timestamps must be valid datetime format",
            check_function=self._check_valid_timestamps,
            severity="error"
        ))
        
        self.add_rule(DataQualityRule(
            name="positive_power_values",
            description="Power values must be positive",
            check_function=self._check_positive_power_values,
            severity="warning"
        ))
        
        # Consistency rules
        self.add_rule(DataQualityRule(
            name="timestamp_ordering",
            description="Timestamps must be in chronological order",
            check_function=self._check_timestamp_ordering,
            severity="warning"
        ))
        
        self.add_rule(DataQualityRule(
            name="reasonable_power_range",
            description="Power values must be within reasonable range",
            check_function=self._check_power_range,
            severity="warning"
        ))
        
        # Uniqueness rules
        self.add_rule(DataQualityRule(
            name="no_duplicate_records",
            description="No duplicate records allowed",
            check_function=self._check_no_duplicates,
            severity="error"
        ))
    
    def add_rule(self, rule: DataQualityRule):
        """Add custom data quality rule."""
        self.rules.append(rule)
        logger.info(f"Added data quality rule: {rule.name}")
    
    def check_data_quality(self, df: pd.DataFrame) -> List[DataQualityResult]:
        """Check data quality against all rules."""
        results = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                passed, message, details = rule.check_function(df)
                result = DataQualityResult(
                    rule_name=rule.name,
                    passed=passed,
                    message=message,
                    severity=rule.severity,
                    timestamp=datetime.utcnow(),
                    details=details
                )
                results.append(result)
                
                if not passed:
                    logger.warning(f"Data quality rule failed: {rule.name} - {message}")
                
            except Exception as e:
                logger.error(f"Error checking rule {rule.name}: {e}")
                result = DataQualityResult(
                    rule_name=rule.name,
                    passed=False,
                    message=f"Rule check failed: {str(e)}",
                    severity="error",
                    timestamp=datetime.utcnow()
                )
                results.append(result)
        
        return results
    
    def _check_no_null_timestamps(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check for null timestamps."""
        null_count = df['timestamp'].isnull().sum()
        if null_count > 0:
            return False, f"Found {null_count} null timestamps", {"null_count": null_count}
        return True, "No null timestamps found", {}
    
    def _check_no_null_power_values(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check for null power values."""
        null_count = df['power_kw'].isnull().sum()
        if null_count > 0:
            return False, f"Found {null_count} null power values", {"null_count": null_count}
        return True, "No null power values found", {}
    
    def _check_valid_timestamps(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check timestamp format validity."""
        try:
            pd.to_datetime(df['timestamp'])
            return True, "All timestamps are valid", {}
        except Exception as e:
            return False, f"Invalid timestamp format: {str(e)}", {"error": str(e)}
    
    def _check_positive_power_values(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check for positive power values."""
        negative_count = (df['power_kw'] < 0).sum()
        if negative_count > 0:
            return False, f"Found {negative_count} negative power values", {"negative_count": negative_count}
        return True, "All power values are positive", {}
    
    def _check_timestamp_ordering(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check timestamp ordering."""
        if len(df) <= 1:
            return True, "Not enough data to check ordering", {}
        
        timestamps = pd.to_datetime(df['timestamp']).sort_values()
        is_ordered = timestamps.is_monotonic_increasing
        if not is_ordered:
            return False, "Timestamps are not in chronological order", {"is_ordered": False}
        return True, "Timestamps are in chronological order", {}
    
    def _check_power_range(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check power values are within reasonable range."""
        min_power = df['power_kw'].min()
        max_power = df['power_kw'].max()
        
        # Reasonable range: 0-10000 kW
        if min_power < 0 or max_power > 10000:
            return False, f"Power values outside reasonable range: {min_power}-{max_power} kW", {
                "min_power": min_power,
                "max_power": max_power
            }
        return True, f"Power values within reasonable range: {min_power}-{max_power} kW", {
            "min_power": min_power,
            "max_power": max_power
        }
    
    def _check_no_duplicates(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check for duplicate records."""
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            return False, f"Found {duplicate_count} duplicate records", {"duplicate_count": duplicate_count}
        return True, "No duplicate records found", {}

class DataLineageTracker:
    """Track data lineage and transformations."""
    
    def __init__(self):
        self.lineage = {}
    
    def record_transformation(
        self, 
        source_id: str, 
        target_id: str, 
        transformation_type: str,
        parameters: Dict[str, Any] = None
    ):
        """Record data transformation."""
        if source_id not in self.lineage:
            self.lineage[source_id] = []
        
        self.lineage[source_id].append({
            "target_id": target_id,
            "transformation_type": transformation_type,
            "parameters": parameters or {},
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Recorded transformation: {source_id} -> {target_id} ({transformation_type})")
    
    def get_lineage(self, data_id: str) -> List[Dict[str, Any]]:
        """Get lineage for specific data."""
        return self.lineage.get(data_id, [])
    
    def get_full_lineage(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get complete lineage information."""
        return self.lineage

class DataRetentionManager:
    """Manage data retention policies."""
    
    def __init__(self):
        self.retention_policies = {
            "raw_data": timedelta(days=365),  # 1 year
            "processed_data": timedelta(days=180),  # 6 months
            "model_artifacts": timedelta(days=90),  # 3 months
            "logs": timedelta(days=30),  # 30 days
            "metrics": timedelta(days=90),  # 3 months
        }
    
    def should_retain(self, data_type: str, created_at: datetime) -> bool:
        """Check if data should be retained based on policy."""
        if data_type not in self.retention_policies:
            return True  # Retain if no policy defined
        
        retention_period = self.retention_policies[data_type]
        return datetime.utcnow() - created_at < retention_period
    
    def get_expired_data(self, data_type: str, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get list of expired data for cleanup."""
        if data_type not in self.retention_policies:
            return []
        
        retention_period = self.retention_policies[data_type]
        cutoff_date = datetime.utcnow() - retention_period
        
        expired = []
        for item in data_list:
            if 'created_at' in item:
                created_at = datetime.fromisoformat(item['created_at'])
                if created_at < cutoff_date:
                    expired.append(item)
        
        return expired
    
    def cleanup_expired_data(self, data_type: str, data_path: str) -> int:
        """Clean up expired data files."""
        if not os.path.exists(data_path):
            return 0
        
        expired_files = []
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            if os.path.isfile(file_path):
                created_at = datetime.fromtimestamp(os.path.getctime(file_path))
                if not self.should_retain(data_type, created_at):
                    expired_files.append(file_path)
        
        # Remove expired files
        for file_path in expired_files:
            try:
                os.remove(file_path)
                logger.info(f"Removed expired file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove file {file_path}: {e}")
        
        return len(expired_files)

class DataPrivacyManager:
    """Manage data privacy and compliance."""
    
    def __init__(self):
        self.sensitive_fields = ['machine_id', 'timestamp', 'power_kw']
        self.anonymization_rules = {
            'machine_id': self._anonymize_machine_id,
            'timestamp': self._anonymize_timestamp,
        }
    
    def anonymize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize sensitive data."""
        anonymized_df = df.copy()
        
        for field, rule in self.anonymization_rules.items():
            if field in anonymized_df.columns:
                anonymized_df[field] = rule(anonymized_df[field])
        
        return anonymized_df
    
    def _anonymize_machine_id(self, series: pd.Series) -> pd.Series:
        """Anonymize machine IDs."""
        # Hash machine IDs to anonymize them
        return series.apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8])
    
    def _anonymize_timestamp(self, series: pd.Series) -> pd.Series:
        """Anonymize timestamps by rounding to hour."""
        timestamps = pd.to_datetime(series)
        return timestamps.dt.floor('H')
    
    def check_compliance(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check data compliance requirements."""
        compliance = {
            'has_consent': True,  # Assume consent is given
            'data_minimization': len(df.columns) <= 10,  # Limit columns
            'purpose_limitation': True,  # Assume purpose is limited
            'storage_limitation': True,  # Assume storage is limited
        }
        
        return compliance

class DataBackupManager:
    """Manage data backup and recovery."""
    
    def __init__(self, backup_path: str = "backups"):
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(exist_ok=True)
    
    def create_backup(self, data: pd.DataFrame, backup_name: str) -> str:
        """Create data backup."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_path / f"{backup_name}_{timestamp}.parquet"
        
        try:
            data.to_parquet(backup_file, compression='snappy')
            logger.info(f"Created backup: {backup_file}")
            return str(backup_file)
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_file: str) -> pd.DataFrame:
        """Restore data from backup."""
        try:
            df = pd.read_parquet(backup_file)
            logger.info(f"Restored backup: {backup_file}")
            return df
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise
    
    def list_backups(self, backup_name: str = None) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        pattern = f"{backup_name}_*.parquet" if backup_name else "*.parquet"
        
        for backup_file in self.backup_path.glob(pattern):
            stat = backup_file.stat()
            backups.append({
                'filename': backup_file.name,
                'path': str(backup_file),
                'size': stat.st_size,
                'created_at': datetime.fromtimestamp(stat.st_ctime),
                'modified_at': datetime.fromtimestamp(stat.st_mtime)
            })
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)
    
    def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up old backups."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        removed_count = 0
        
        for backup_file in self.backup_path.glob("*.parquet"):
            created_at = datetime.fromtimestamp(backup_file.stat().st_ctime)
            if created_at < cutoff_date:
                try:
                    backup_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed old backup: {backup_file}")
                except Exception as e:
                    logger.error(f"Failed to remove backup {backup_file}: {e}")
        
        return removed_count

# Global instances
data_quality_checker = DataQualityChecker()
data_lineage_tracker = DataLineageTracker()
data_retention_manager = DataRetentionManager()
data_privacy_manager = DataPrivacyManager()
data_backup_manager = DataBackupManager()

class DataGovernanceOrchestrator:
    """Orchestrate all data governance activities."""
    
    def __init__(self):
        self.quality_checker = data_quality_checker
        self.lineage_tracker = data_lineage_tracker
        self.retention_manager = data_retention_manager
        self.privacy_manager = data_privacy_manager
        self.backup_manager = data_backup_manager
    
    async def process_new_data(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """Process new data through governance pipeline."""
        results = {
            'source': source,
            'timestamp': datetime.utcnow().isoformat(),
            'quality_checks': [],
            'compliance_checks': {},
            'backup_created': False,
            'lineage_recorded': False
        }
        
        # Quality checks
        quality_results = self.quality_checker.check_data_quality(df)
        results['quality_checks'] = [
            {
                'rule': r.rule_name,
                'passed': r.passed,
                'message': r.message,
                'severity': r.severity
            }
            for r in quality_results
        ]
        
        # Compliance checks
        results['compliance_checks'] = self.privacy_manager.check_compliance(df)
        
        # Create backup
        try:
            backup_path = self.backup_manager.create_backup(df, source)
            results['backup_created'] = True
            results['backup_path'] = backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
        
        # Record lineage
        data_id = f"{source}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.lineage_tracker.record_transformation(
            source_id=source,
            target_id=data_id,
            transformation_type="data_ingestion",
            parameters={"rows": len(df), "columns": len(df.columns)}
        )
        results['lineage_recorded'] = True
        
        return results
    
    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data across all data types."""
        cleanup_results = {}
        
        for data_type in self.retention_manager.retention_policies.keys():
            try:
                # This would be implemented based on your specific data storage
                # For now, return mock results
                cleanup_results[data_type] = 0
            except Exception as e:
                logger.error(f"Failed to cleanup {data_type}: {e}")
                cleanup_results[data_type] = 0
        
        return cleanup_results

# Global orchestrator
data_governance_orchestrator = DataGovernanceOrchestrator()
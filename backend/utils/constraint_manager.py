"""
Constraint management for causal graph editing.

This module provides functionality to save, load, and apply
domain knowledge constraints to causal graphs.
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from backend.core.logging import get_logger

logger = get_logger(__name__)


class ConstraintManager:
    """
    Manages causal graph constraints (domain knowledge).
    
    Constraints allow users to enforce, forbid, or delete edges
    in the learned causal graph based on expert knowledge.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize ConstraintManager.
        
        Args:
            base_dir: Base directory for saving constraints. 
                     If None, uses PROJECT_ROOT/backend/session_constraints
        """
        if base_dir is None:
            from backend.utils import PROJECT_ROOT
            base_dir = PROJECT_ROOT / "backend" / "session_constraints"
        
        self.base_dir = Path(base_dir)
        self._ensure_base_dir()
    
    def _ensure_base_dir(self) -> None:
        """Create base directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Constraint storage directory: {self.base_dir}")
    
    def _get_constraint_file(self, session_id: str) -> Path:
        """Get file path for session constraints."""
        return self.base_dir / f"{session_id}.json"
    
    def load_constraints(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load constraints for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of constraint dictionaries
        """
        constraint_file = self._get_constraint_file(session_id)
        
        if not constraint_file.exists():
            logger.debug(f"No constraints found for session {session_id}")
            return []
        
        try:
            with open(constraint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            constraints = data.get('constraints', [])
            logger.info(f"Loaded {len(constraints)} constraints for session {session_id}")
            return constraints
            
        except Exception as e:
            logger.error(f"Failed to load constraints", session_id=session_id, error=str(e))
            return []
    
    def save_constraints(self, session_id: str, constraints: List[Dict[str, Any]]) -> bool:
        """
        Save constraints for a session.
        
        Args:
            session_id: Session identifier
            constraints: List of constraint dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            constraint_file = self._get_constraint_file(session_id)
            
            data = {
                'session_id': session_id,
                'constraints': constraints,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(constraint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(constraints)} constraints for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save constraints", session_id=session_id, error=str(e))
            return False
    
    def add_constraint(
        self,
        session_id: str,
        from_skill: str,
        to_skill: str,
        constraint_type: str,
        value: Optional[float] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a new constraint.
        
        Args:
            session_id: Session identifier
            from_skill: Source skill name
            to_skill: Target skill name
            constraint_type: Type of constraint ('required', 'forbidden', 'deleted')
            value: Optional value for 'required' constraints
            user_id: Optional user identifier
            
        Returns:
            The created constraint dictionary
        """
        constraints = self.load_constraints(session_id)
        
        # Generate unique ID
        constraint_id = str(uuid.uuid4())
        
        # Create constraint object
        constraint = {
            'id': constraint_id,
            'from_skill': from_skill,
            'to_skill': to_skill,
            'constraint_type': constraint_type,
            'value': value if constraint_type == 'required' else None,
            'created_at': datetime.now().isoformat(),
            'created_by': user_id
        }
        
        # Check for duplicate
        for existing in constraints:
            if (existing['from_skill'] == from_skill and 
                existing['to_skill'] == to_skill):
                # Update existing constraint
                existing.update(constraint)
                self.save_constraints(session_id, constraints)
                logger.info(f"Updated existing constraint: {from_skill} -> {to_skill}")
                return existing
        
        # Add new constraint
        constraints.append(constraint)
        self.save_constraints(session_id, constraints)
        logger.info(f"Added constraint: {from_skill} -> {to_skill} ({constraint_type})")
        
        return constraint
    
    def delete_constraint(self, session_id: str, constraint_id: str) -> bool:
        """
        Delete a constraint by ID.
        
        Args:
            session_id: Session identifier
            constraint_id: Constraint identifier
            
        Returns:
            True if deleted, False if not found
        """
        constraints = self.load_constraints(session_id)
        
        # Find and remove constraint
        for i, constraint in enumerate(constraints):
            if constraint['id'] == constraint_id:
                removed = constraints.pop(i)
                self.save_constraints(session_id, constraints)
                logger.info(
                    f"Deleted constraint: {removed['from_skill']} -> {removed['to_skill']}"
                )
                return True
        
        logger.warning(f"Constraint not found: {constraint_id}")
        return False
    
    def apply_constraints_to_matrix(
        self,
        adj_matrix: pd.DataFrame,
        constraints: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Apply constraints to an adjacency matrix.
        
        Args:
            adj_matrix: Adjacency matrix (from -> to, row=from, col=to)
            constraints: List of constraints to apply
            
        Returns:
            Modified adjacency matrix
        """
        result = adj_matrix.copy()
        applied_count = 0
        skipped_count = 0
        
        for constraint in constraints:
            from_skill = constraint['from_skill']
            to_skill = constraint['to_skill']
            c_type = constraint['constraint_type']
            
            # Skip if skills don't exist in matrix
            if from_skill not in result.index or to_skill not in result.columns:
                logger.warning(
                    f"Skill not in matrix: {from_skill} -> {to_skill}, skipping"
                )
                skipped_count += 1
                continue
            
            # Apply constraint
            if c_type == 'required':
                # Force edge to exist with specified value
                value = constraint.get('value', 0.5)
                result.loc[from_skill, to_skill] = value
                applied_count += 1
                
            elif c_type in ['forbidden', 'deleted']:
                # Force edge to not exist
                result.loc[from_skill, to_skill] = 0.0
                applied_count += 1
        
        logger.info(
            f"Applied {applied_count} constraints, skipped {skipped_count}"
        )
        return result
    
    def build_prior_knowledge_matrix(
        self,
        feature_names: List[str],
        constraints: List[Dict[str, Any]]
    ) -> 'np.ndarray':
        """
        制約リストからLiNGAM用のprior_knowledge行列を構築
        
        Args:
            feature_names: スキル名のリスト（順序はskill_matrixと同じ）
            constraints: 制約リスト
            
        Returns:
            prior_knowledge行列 (n x n)
            -1: データから学習（デフォルト）
             0: 因果関係なし（禁止）
             1: 因果関係あり（必須）
        """
        import numpy as np
        
        n = len(feature_names)
        prior_knowledge = np.full((n, n), -1, dtype=int)  # デフォルト: -1 (不明)
        
       # スキル名→インデックスのマップ
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        applied_count = 0
        skipped_count = 0
        
        for constraint in constraints:
            from_skill = constraint.get('from_skill')
            to_skill = constraint.get('to_skill')
            c_type = constraint.get('constraint_type')
            
            if from_skill not in name_to_idx or to_skill not in name_to_idx:
                logger.debug(f"Constraint skill not found: {from_skill} -> {to_skill}")
                skipped_count += 1
                continue
            
            i = name_to_idx[from_skill]
            j = name_to_idx[to_skill]
            
            if c_type == 'required':
                prior_knowledge[i, j] = 1  # 必須
                applied_count += 1
            elif c_type in ['forbidden', 'deleted']:
                prior_knowledge[i, j] = 0  # 禁止
                applied_count += 1
        
        logger.info(
            f"Built prior_knowledge matrix: {applied_count} constraints applied, "
            f"{skipped_count} skipped"
        )
        
        return prior_knowledge


# Singleton instance
constraint_manager = ConstraintManager()

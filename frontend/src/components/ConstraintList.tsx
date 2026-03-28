/**
 * Constraint List Component
 * 
 * Displays list of constraints for a session with delete functionality
 */

import React from 'react';
import { X, CheckCircle, XCircle } from 'lucide-react';

interface Constraint {
    id: string;
    from_skill: string;
    to_skill: string;
    constraint_type: string;
    value?: number;
    created_at: string;
}

interface ConstraintListProps {
    constraints: Constraint[];
    onDelete: (constraintId: string) => void;
}

export const ConstraintList: React.FC<ConstraintListProps> = ({
    constraints,
    onDelete,
}) => {
    if (constraints.length === 0) {
        return (
            <div className="text-center py-8 text-gray-500">
                制約が設定されていません
            </div>
        );
    }

    const getConstraintIcon = (type: string) => {
        switch (type) {
            case 'required':
                return <CheckCircle size={16} className="text-green-600" />;
            case 'forbidden':
            case 'deleted':
                return <XCircle size={16} className="text-red-600" />;
            default:
                return null;
        }
    };

    const getConstraintLabel = (type: string) => {
        switch (type) {
            case 'required':
                return '必須';
            case 'forbidden':
                return '禁止';
            case 'deleted':
                return '削除';
            default:
                return type;
        }
    };

    const getConstraintColor = (type: string) => {
        switch (type) {
            case 'required':
                return 'bg-green-50 border-green-200';
            case 'forbidden':
            case 'deleted':
                return 'bg-red-50 border-red-200';
            default:
                return 'bg-gray-50 border-gray-200';
        }
    };

    return (
        <div className="space-y-2">
            {constraints.map((constraint) => (
                <div
                    key={constraint.id}
                    className={`p-3 rounded-lg border ${getConstraintColor(
                        constraint.constraint_type
                    )} flex items-center justify-between`}
                >
                    <div className="flex items-center gap-3 flex-1">
                        {getConstraintIcon(constraint.constraint_type)}
                        <div className="flex-1">
                            <div className="text-sm text-gray-900">
                                <span className="font-medium">{constraint.from_skill}</span>
                                <span className="mx-2 text-gray-500">→</span>
                                <span className="font-medium">{constraint.to_skill}</span>
                            </div>
                            <div className="text-xs text-gray-600 mt-1">
                                {getConstraintLabel(constraint.constraint_type)}
                                {constraint.value !== undefined &&
                                    ` (値: ${constraint.value})`}
                            </div>
                        </div>
                    </div>
                    <button
                        onClick={() => onDelete(constraint.id)}
                        className="p-1 hover:bg-gray-200 rounded transition-colors"
                        title="制約を削除"
                    >
                        <X size={16} className="text-gray-600" />
                    </button>
                </div>
            ))}
        </div>
    );
};

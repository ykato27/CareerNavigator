/**
 * Constraint Management Panel Component
 * 
 * Manages causal graph constraints for a session
 * Provides UI for viewing, adding, and deleting constraints
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Edit3, ChevronDown, ChevronUp, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/constants';
import { ConstraintList } from './ConstraintList';
import { AddConstraintForm } from './AddConstraintForm';

interface Constraint {
    id: string;
    from_skill: string;
    to_skill: string;
    constraint_type: string;
    value?: number;
    created_at: string;
}

interface ConstraintManagementPanelProps {
    sessionId: string | null;
    modelId: string;
    onConstraintsApplied?: () => void;
}

export const ConstraintManagementPanel: React.FC<ConstraintManagementPanelProps> = ({
    sessionId,
    modelId,
    onConstraintsApplied,
}) => {
    const [showPanel, setShowPanel] = useState(false);
    const [constraints, setConstraints] = useState<Constraint[]>([]);
    const [loadingConstraints, setLoadingConstraints] = useState(false);
    const [addingConstraint, setAddingConstraint] = useState(false);
    const [applyingConstraints, setApplyingConstraints] = useState(false);
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');
    const [allSkills, setAllSkills] = useState<string[]>([]);

    // Fetch constraints when panel is opened
    useEffect(() => {
        if (showPanel && sessionId) {
            fetchConstraints();
        }
    }, [showPanel, sessionId]);

    // Load available skills from session
    useEffect(() => {
        if (sessionId && modelId) {
            loadSkills();
        }
    }, [sessionId, modelId]);

    const loadSkills = async () => {
        if (!sessionId || !modelId) return;

        try {
            // Get skills from the model's skill matrix
            const response = await axios.get(`${API_BASE_URL}/api/skills/${modelId}`);
            setAllSkills(response.data.skills || []);
        } catch (err: any) {
            console.error('Failed to load skills from model:', err);

            // Fallback: Use basic hardcoded skills if API doesn't exist yet
            setAllSkills([
                'プログラミング基礎',
                'データベース設計',
                '機械学習',
                'Python',
                'SQL',
                'Web開発',
                'データ分析',
                'プロジェクト管理',
            ]);
        }
    };

    const fetchConstraints = async () => {
        if (!sessionId) return;

        setLoadingConstraints(true);
        setError('');

        try {
            const response = await axios.get(`${API_BASE_URL}${API_ENDPOINTS.CONSTRAINTS(sessionId)}`);
            setConstraints(response.data.constraints || []);
        } catch (err: any) {
            setError(err.response?.data?.detail || '制約の取得に失敗しました');
        } finally {
            setLoadingConstraints(false);
        }
    };

    const handleAddConstraint = async (
        fromSkill: string,
        toSkill: string,
        constraintType: string,
        value?: number
    ) => {
        if (!sessionId) return;

        setAddingConstraint(true);
        setError('');
        setSuccessMessage('');

        try {
            await axios.post(`${API_BASE_URL}${API_ENDPOINTS.CONSTRAINTS(sessionId)}`, {
                from_skill: fromSkill,
                to_skill: toSkill,
                constraint_type: constraintType,
                value: value,
            });

            setSuccessMessage('制約を追加しました');
            await fetchConstraints();

            // Clear success message after 3 seconds
            setTimeout(() => setSuccessMessage(''), 3000);
        } catch (err: any) {
            setError(err.response?.data?.detail || '制約の追加に失敗しました');
        } finally {
            setAddingConstraint(false);
        }
    };

    const handleDeleteConstraint = async (constraintId: string) => {
        if (!sessionId) return;

        setError('');
        setSuccessMessage('');

        try {
            await axios.delete(`${API_BASE_URL}${API_ENDPOINTS.CONSTRAINT_DELETE(sessionId, constraintId)}`);

            setSuccessMessage('制約を削除しました');
            await fetchConstraints();

            // Clear success message after 3 seconds
            setTimeout(() => setSuccessMessage(''), 3000);
        } catch (err: any) {
            setError(err.response?.data?.detail || '制約の削除に失敗しました');
        }
    };

    const handleApplyConstraints = async () => {
        if (!sessionId || !modelId) {
            setError('セッションIDまたはモデルIDが設定されていません');
            return;
        }

        setApplyingConstraints(true);
        setError('');
        setSuccessMessage('');

        try {
            const response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.CONSTRAINTS_APPLY(sessionId)}`, {
                model_id: modelId,
            });

            const { applied_count, skipped_count } = response.data;
            setSuccessMessage(`制約を適用しました（適用: ${applied_count}件、スキップ: ${skipped_count}件）`);

            // Call callback if provided
            if (onConstraintsApplied) {
                onConstraintsApplied();
            }
        } catch (err: any) {
            setError(err.response?.data?.detail || '制約の適用に失敗しました');
        } finally {
            setApplyingConstraints(false);
        }
    };

    if (!sessionId || !modelId) {
        return null;
    }

    return (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
            <button
                onClick={() => setShowPanel(!showPanel)}
                className="w-full flex items-center justify-between text-left"
                type="button"
            >
                <div className="flex items-center gap-2">
                    <Edit3 size={20} className="text-[#00A968]" />
                    <h2 className="text-lg font-semibold text-gray-800">制約管理（オプション）</h2>
                </div>
                {showPanel ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
            </button>

            {showPanel && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                        <div className="flex items-start gap-2">
                            <AlertCircle size={16} className="text-blue-600 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-blue-800">
                                因果グラフに制約を追加して、ドメイン知識を反映できます。
                                制約を追加後、「制約を適用して再学習」ボタンをクリックしてください。
                            </p>
                        </div>
                    </div>

                    {/* Error Message */}
                    {error && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 flex items-start gap-2">
                            <AlertCircle size={16} className="text-red-600 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-red-800">{error}</p>
                        </div>
                    )}

                    {/* Success Message */}
                    {successMessage && (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4 flex items-start gap-2">
                            <CheckCircle size={16} className="text-green-600 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-green-800">{successMessage}</p>
                        </div>
                    )}

                    {/* Constraints List */}
                    <div className="mb-6">
                        <h3 className="text-sm font-semibold text-gray-800 mb-3">現在の制約</h3>
                        {loadingConstraints ? (
                            <div className="flex justify-center py-8">
                                <Loader2 size={24} className="animate-spin text-[#00A968]" />
                            </div>
                        ) : (
                            <ConstraintList constraints={constraints} onDelete={handleDeleteConstraint} />
                        )}
                    </div>

                    {/* Add Constraint Form */}
                    <AddConstraintForm
                        skills={allSkills}
                        onAdd={handleAddConstraint}
                        loading={addingConstraint}
                    />

                    {/* Apply Constraints / Reset Button */}
                    <div className="mt-6 pt-6 border-t border-gray-200">
                        <button
                            onClick={handleApplyConstraints}
                            disabled={applyingConstraints}
                            className="w-full bg-[#00A968] text-white py-3 rounded-md font-medium hover:bg-[#008F58] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {applyingConstraints ? (
                                <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                        <Loader2 size={20} className="animate-spin" />
                                        <span>制約を適用して再学習中...（30-90秒）</span>
                                    </div>
                                    {/* プログレスバー */}
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                        <div className="bg-[#00A968] h-2 rounded-full animate-pulse" style={{ width: '100%' }} />
                                    </div>
                                </div>
                            ) : (
                                <>
                                    <Edit3 size={20} />
                                    {constraints.length > 0 ? '制約を適用して再学習' : '制約をリセットして再学習'}
                                </>
                            )}
                        </button>
                        <p className="text-xs text-gray-500 text-center mt-2">
                            {constraints.length > 0
                                ? '※ 制約を適用すると因果グラフが更新され、推薦結果に反映されます'
                                : '※ 制約を削除した場合、このボタンで元の因果グラフに戻します'}
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

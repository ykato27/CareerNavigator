/**
 * Simple Constraint Add Form Component
 * 
 * Allows users to add constraints by selecting skills and constraint type
 */

import React, { useState } from 'react';
import { Plus } from 'lucide-react';

interface AddConstraintFormProps {
    skills: string[];
    onAdd: (fromSkill: string, toSkill: string, constraintType: string, value?: number) => void;
    loading?: boolean;
}

export const AddConstraintForm: React.FC<AddConstraintFormProps> = ({
    skills,
    onAdd,
    loading = false,
}) => {
    const [fromSkill, setFromSkill] = useState('');
    const [toSkill, setToSkill] = useState('');
    const [constraintType, setConstraintType] = useState<string>('required');
    const [value, setValue] = useState(0.5);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (!fromSkill || !toSkill) {
            alert('両方のスキルを選択してください');
            return;
        }

        if (fromSkill === toSkill) {
            alert('異なるスキルを選択してください');
            return;
        }

        onAdd(fromSkill, toSkill, constraintType, constraintType === 'required' ? value : undefined);

        // Reset form
        setFromSkill('');
        setToSkill('');
        setConstraintType('required');
        setValue(0.5);
    };

    return (
        <form onSubmit={handleSubmit} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <h3 className="text-sm font-semibold text-gray-800 mb-3">新しい制約を追加</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                {/* From Skill */}
                <div>
                    <label className="block text-sm text-gray-700 mb-2">
                        From スキル
                    </label>
                    <select
                        value={fromSkill}
                        onChange={(e) => setFromSkill(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#00A968] focus:border-transparent"
                        disabled={loading}
                    >
                        <option value="">選択してください</option>
                        {skills.map((skill) => (
                            <option key={`from-${skill}`} value={skill}>
                                {skill}
                            </option>
                        ))}
                    </select>
                </div>

                {/* To Skill */}
                <div>
                    <label className="block text-sm text-gray-700 mb-2">
                        To スキル
                    </label>
                    <select
                        value={toSkill}
                        onChange={(e) => setToSkill(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#00A968] focus:border-transparent"
                        disabled={loading}
                    >
                        <option value="">選択してください</option>
                        {skills.map((skill) => (
                            <option key={`to-${skill}`} value={skill}>
                                {skill}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Constraint Type */}
            <div className="mb-4">
                <label className="block text-sm text-gray-700 mb-2">
                    制約タイプ
                </label>
                <div className="space-y-2">
                    <label className="flex items-center">
                        <input
                            type="radio"
                            name="constraintType"
                            value="required"
                            checked={constraintType === 'required'}
                            onChange={(e) => setConstraintType(e.target.value)}
                            className="mr-2"
                            disabled={loading}
                        />
                        <span className="text-sm text-gray-700">必須（この関係は必ず存在）</span>
                    </label>
                    <label className="flex items-center">
                        <input
                            type="radio"
                            name="constraintType"
                            value="forbidden"
                            checked={constraintType === 'forbidden'}
                            onChange={(e) => setConstraintType(e.target.value)}
                            className="mr-2"
                            disabled={loading}
                        />
                        <span className="text-sm text-gray-700">禁止（この関係は存在しない）</span>
                    </label>
                </div>
            </div>

            {/* Value for required constraints */}
            {constraintType === 'required' && (
                <div className="mb-4">
                    <label className="block text-sm text-gray-700 mb-2">
                        因果効果の値: {value.toFixed(2)}
                    </label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={value}
                        onChange={(e) => setValue(parseFloat(e.target.value))}
                        className="w-full"
                        disabled={loading}
                    />
                    <p className="text-xs text-gray-500 mt-1">
                        0.1〜1.0の範囲で設定してください
                    </p>
                </div>
            )}

            <button
                type="submit"
                disabled={loading || !fromSkill || !toSkill}
                className="w-full px-4 py-2 bg-[#00A968] text-white rounded-md hover:bg-[#008F58] disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
            >
                <Plus size={16} />
                制約を追加
            </button>
        </form>
    );
};

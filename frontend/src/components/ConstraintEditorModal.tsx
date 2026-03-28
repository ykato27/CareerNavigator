/**
 * Constraint Editor Modal Component
 * 
 * Allows users to set constraints on causal graph edges:
 * - required: This edge must exist
 * - forbidden: This edge must not exist
 * - deleted: Remove this edge
 */

import React, { useState } from 'react';
import { X } from 'lucide-react';

interface ConstraintEditorModalProps {
    isOpen: boolean;
    onClose: () => void;
    fromSkill: string;
    toSkill: string;
    currentType: 'free' | 'required' | 'forbidden' | 'deleted';
    onSave: (constraintType: string, value?: number) => void;
}

export const ConstraintEditorModal: React.FC<ConstraintEditorModalProps> = ({
    isOpen,
    onClose,
    fromSkill,
    toSkill,
    currentType,
    onSave,
}) => {
    const [selectedType, setSelectedType] = useState<string>(currentType);
    const [value, setValue] = useState<number>(0.5);

    if (!isOpen) return null;

    const handleSave = () => {
        if (selectedType === 'free') {
            // TODO: Delete constraint if exists
            onClose();
            return;
        }

        onSave(selectedType, selectedType === 'required' ? value : undefined);
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
                {/* Header */}
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-bold text-gray-900">因果関係の制約設定</h2>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-gray-600"
                    >
                        <X size={24} />
                    </button>
                </div>

                {/* Edge Info */}
                <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-600 mb-1">From</div>
                    <div className="font-medium text-gray-900 mb-3">{fromSkill}</div>
                    <div className="text-sm text-gray-600 mb-1">To</div>
                    <div className="font-medium text-gray-900">{toSkill}</div>
                </div>

                {/* Constraint Type Selection */}
                <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                        制約タイプ
                    </label>

                    <div className="space-y-2">
                        {/* Free */}
                        <label className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                            <input
                                type="radio"
                                name="constraintType"
                                value="free"
                                checked={selectedType === 'free'}
                                onChange={(e) => setSelectedType(e.target.value)}
                                className="mr-3"
                            />
                            <div>
                                <div className="font-medium text-gray-900">自由（学習結果のまま）</div>
                                <div className="text-sm text-gray-500">データから学習された値を使用</div>
                            </div>
                        </label>

                        {/* Required */}
                        <label className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                            <input
                                type="radio"
                                name="constraintType"
                                value="required"
                                checked={selectedType === 'required'}
                                onChange={(e) => setSelectedType(e.target.value)}
                                className="mr-3"
                            />
                            <div className="flex-1">
                                <div className="font-medium text-gray-900">必須（この関係は必ず存在）</div>
                                <div className="text-sm text-gray-500">専門知識に基づき強制的に設定</div>
                                {selectedType === 'required' && (
                                    <div className="mt-2">
                                        <label className="block text-xs text-gray-600 mb-1">因果効果の値</label>
                                        <input
                                            type="number"
                                            value={value}
                                            onChange={(e) => setValue(parseFloat(e.target.value))}
                                            min="0"
                                            max="1"
                                            step="0.1"
                                            className="w-full px-3 py-1 border rounded text-sm"
                                        />
                                    </div>
                                )}
                            </div>
                        </label>

                        {/* Forbidden */}
                        <label className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                            <input
                                type="radio"
                                name="constraintType"
                                value="forbidden"
                                checked={selectedType === 'forbidden'}
                                onChange={(e) => setSelectedType(e.target.value)}
                                className="mr-3"
                            />
                            <div>
                                <div className="font-medium text-gray-900">禁止（この関係は存在しない）</div>
                                <div className="text-sm text-gray-500">因果関係を強制的に削除</div>
                            </div>
                        </label>
                    </div>
                </div>

                {/* Actions */}
                <div className="flex justify-end space-x-3">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                    >
                        キャンセル
                    </button>
                    <button
                        onClick={handleSave}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                    >
                        保存
                    </button>
                </div>
            </div>
        </div>
    );
};

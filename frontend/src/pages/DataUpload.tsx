import { useState } from 'react';
import { FileText, CheckCircle, AlertCircle, X, Plus } from 'lucide-react';
import axios from 'axios';
import { clsx } from 'clsx';

const FileUploadCard = ({
    label,
    fileKey,
    files,
    onFileSelect,
    onFileDelete
}: {
    label: string,
    fileKey: string,
    files: File[],
    onFileSelect: (key: string, file: File) => void,
    onFileDelete: (key: string, index: number) => void
}) => {
    return (
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm hover:border-green-500 transition-colors">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-gray-700">{label}</h3>
                {files.length > 0 ? (
                    <CheckCircle className="text-green-500" size={20} />
                ) : (
                    <div className="w-5 h-5 rounded-full border-2 border-gray-300"></div>
                )}
            </div>

            {/* Uploaded files list */}
            {files.length > 0 && (
                <div className="mb-3 space-y-2 max-h-32 overflow-y-auto">
                    {files.map((file, index) => (
                        <div key={index} className="flex items-center gap-2 bg-green-50 border border-green-200 rounded-lg p-2">
                            <FileText className="text-green-600 flex-shrink-0" size={16} />
                            <div className="flex-1 min-w-0">
                                <p className="text-xs text-gray-700 font-medium truncate">{file.name}</p>
                                <p className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                            </div>
                            <button
                                onClick={() => onFileDelete(fileKey, index)}
                                className="text-red-500 hover:text-red-700 flex-shrink-0"
                                title="削除"
                            >
                                <X size={16} />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Upload area */}
            <div className="relative">
                <input
                    type="file"
                    accept=".csv"
                    multiple
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    onChange={(e) => {
                        if (e.target.files) {
                            Array.from(e.target.files).forEach(file => {
                                onFileSelect(fileKey, file);
                            });
                            e.target.value = ''; // Reset input to allow re-selecting same file
                        }
                    }}
                />
                <div className={clsx(
                    "border-2 border-dashed rounded-lg p-4 flex flex-col items-center justify-center text-center h-24 transition-colors",
                    files.length > 0 ? "border-green-200 bg-green-50" : "border-gray-300 hover:bg-gray-50"
                )}>
                    <Plus className="text-gray-400 mb-1" size={20} />
                    <p className="text-xs text-gray-500">CSVファイルを追加</p>
                </div>
            </div>
        </div>
    );
};

export const DataUpload = () => {
    const [files, setFiles] = useState<Record<string, File[]>>({
        members: [],
        skills: [],
        education: [],
        license: [],
        categories: [],
        acquired: []
    });

    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);

    const handleFileSelect = (key: string, file: File) => {
        setFiles(prev => ({
            ...prev,
            [key]: [...prev[key], file]
        }));
        setError(null);
    };

    const handleFileDelete = (key: string, index: number) => {
        setFiles(prev => ({
            ...prev,
            [key]: prev[key].filter((_, i) => i !== index)
        }));
        setError(null);
    };

    const handleUpload = async () => {
        // Check if all categories have at least one file
        const missingFiles = Object.entries(files).filter(([_, fileList]) => fileList.length === 0).map(([k]) => k);
        if (missingFiles.length > 0) {
            setError(`全てのカテゴリにファイルを選択してください (${missingFiles.length}個未選択)`);
            return;
        }

        setUploading(true);
        setError(null);

        const formData = new FormData();
        Object.entries(files).forEach(([key, fileList]) => {
            fileList.forEach((file, index) => {
                // Append multiple files with indexed names
                formData.append(`${key}[${index}]`, file);
            });
        });

        try {
            const response = await axios.post('http://localhost:8000/api/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            setSuccess(true);

            // Store session_id and upload state for later use
            sessionStorage.setItem('career_session_id', response.data.session_id);
            sessionStorage.setItem('career_data_uploaded', 'true');
            sessionStorage.setItem('career_upload_time', new Date().toISOString());

            console.log("Upload success:", response.data);

        } catch (err: any) {
            const errorMsg = err.response?.data?.detail || "アップロードに失敗しました。サーバーの状態を確認してください。";
            setError(errorMsg);
            console.error(err);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-gray-800">データ読み込み</h2>
                    <p className="text-gray-500 mt-1">6種類のCSVファイルをアップロードしてデータを準備します</p>
                </div>

                <button
                    onClick={handleUpload}
                    disabled={uploading || success}
                    className={clsx(
                        "px-6 py-3 rounded-lg font-bold text-white shadow-sm transition-all flex items-center gap-2",
                        uploading ? "bg-gray-400 cursor-not-allowed" :
                            success ? "bg-green-600 cursor-default" : "bg-primary hover:bg-green-700"
                    )}
                >
                    {uploading ? "読み込み中..." : success ? "読み込み完了" : "データを読み込む"}
                </button>
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
                    <AlertCircle size={20} />
                    {error}
                </div>
            )}

            {success && (
                <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg flex items-center gap-2">
                    <CheckCircle size={20} />
                    データの読み込みが完了しました。メニューから分析機能を利用できます。
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <FileUploadCard label="1. メンバーマスタ" fileKey="members" files={files.members} onFileSelect={handleFileSelect} onFileDelete={handleFileDelete} />
                <FileUploadCard label="2. 力量（スキル）マスタ" fileKey="skills" files={files.skills} onFileSelect={handleFileSelect} onFileDelete={handleFileDelete} />
                <FileUploadCard label="3. 力量（教育）マスタ" fileKey="education" files={files.education} onFileSelect={handleFileSelect} onFileDelete={handleFileDelete} />
                <FileUploadCard label="4. 力量（資格）マスタ" fileKey="license" files={files.license} onFileSelect={handleFileSelect} onFileDelete={handleFileDelete} />
                <FileUploadCard label="5. 力量カテゴリーマスタ" fileKey="categories" files={files.categories} onFileSelect={handleFileSelect} onFileDelete={handleFileDelete} />
                <FileUploadCard label="6. 保有力量データ" fileKey="acquired" files={files.acquired} onFileSelect={handleFileSelect} onFileDelete={handleFileDelete} />
            </div>
        </div>
    );
};

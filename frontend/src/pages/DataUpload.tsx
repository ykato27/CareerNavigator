import { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import axios from 'axios';
import { clsx } from 'clsx';

const FileUploadCard = ({
    label,
    fileKey,
    file,
    onFileSelect
}: {
    label: string,
    fileKey: string,
    file: File | null,
    onFileSelect: (key: string, file: File) => void
}) => {
    return (
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm hover:border-green-500 transition-colors">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-gray-700">{label}</h3>
                {file ? (
                    <CheckCircle className="text-green-500" size={20} />
                ) : (
                    <div className="w-5 h-5 rounded-full border-2 border-gray-300"></div>
                )}
            </div>

            <div className="relative">
                <input
                    type="file"
                    accept=".csv"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    onChange={(e) => {
                        if (e.target.files && e.target.files[0]) {
                            onFileSelect(fileKey, e.target.files[0]);
                        }
                    }}
                />
                <div className={clsx(
                    "border-2 border-dashed rounded-lg p-4 flex flex-col items-center justify-center text-center h-32 transition-colors",
                    file ? "border-green-200 bg-green-50" : "border-gray-300 hover:bg-gray-50"
                )}>
                    {file ? (
                        <>
                            <FileText className="text-green-600 mb-2" size={24} />
                            <p className="text-sm text-gray-700 font-medium truncate w-full px-2">{file.name}</p>
                            <p className="text-xs text-gray-500 mt-1">{(file.size / 1024).toFixed(1)} KB</p>
                        </>
                    ) : (
                        <>
                            <Upload className="text-gray-400 mb-2" size={24} />
                            <p className="text-sm text-gray-500">クリックしてCSVを選択</p>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export const DataUpload = () => {
    const [files, setFiles] = useState<Record<string, File | null>>({
        members: null,
        skills: null,
        education: null,
        license: null,
        categories: null,
        acquired: null
    });

    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);

    const handleFileSelect = (key: string, file: File) => {
        setFiles(prev => ({ ...prev, [key]: file }));
        setError(null);
    };

    const handleUpload = async () => {
        // Check if all files are selected
        const missingFiles = Object.entries(files).filter(([_, f]) => !f).map(([k]) => k);
        if (missingFiles.length > 0) {
            setError(`全てのファイルを選択してください (${missingFiles.length}個未選択)`);
            return;
        }

        setUploading(true);
        setError(null);

        const formData = new FormData();
        Object.entries(files).forEach(([key, file]) => {
            if (file) formData.append(key, file);
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

        } catch (err) {
            setError("アップロードに失敗しました。サーバーの状態を確認してください。");
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
                <FileUploadCard label="1. メンバーマスタ" fileKey="members" file={files.members} onFileSelect={handleFileSelect} />
                <FileUploadCard label="2. 力量（スキル）マスタ" fileKey="skills" file={files.skills} onFileSelect={handleFileSelect} />
                <FileUploadCard label="3. 力量（教育）マスタ" fileKey="education" file={files.education} onFileSelect={handleFileSelect} />
                <FileUploadCard label="4. 力量（資格）マスタ" fileKey="license" file={files.license} onFileSelect={handleFileSelect} />
                <FileUploadCard label="5. 力量カテゴリーマスタ" fileKey="categories" file={files.categories} onFileSelect={handleFileSelect} />
                <FileUploadCard label="6. 保有力量データ" fileKey="acquired" file={files.acquired} onFileSelect={handleFileSelect} />
            </div>
        </div>
    );
};

'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { api } from '@/lib/api';

interface UploadZoneProps {
  onUploadSuccess: (docName: string) => void;
  onToast: (msg: string, type: 'success' | 'error' | 'info') => void;
}

export default function UploadZone({ onUploadSuccess, onToast }: UploadZoneProps) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback(async (files: File[]) => {
    const pdf = files.find(f => f.name.toLowerCase().endsWith('.pdf'));
    if (!pdf) { onToast('Only PDF files are accepted.', 'error'); return; }

    setUploading(true);
    setProgress(20);

    // Simulate progress while uploading
    const timer = setInterval(() => setProgress(p => Math.min(p + 10, 85)), 400);

    try {
      const result = await api.uploadDocument(pdf);
      clearInterval(timer);
      setProgress(100);
      setTimeout(() => { setUploading(false); setProgress(0); }, 500);
      onUploadSuccess(result.document);
      onToast(`✓ Ingested "${result.document}" — ${result.chunks_created} chunks created`, 'success');
    } catch (err: unknown) {
      clearInterval(timer);
      setUploading(false);
      setProgress(0);
      onToast(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error');
    }
  }, [onUploadSuccess, onToast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
    disabled: uploading,
  });

  return (
    <div
      {...getRootProps()}
      className={`upload-zone ${isDragActive ? 'dragover' : ''} ${uploading ? 'uploading' : ''}`}
    >
      <input {...getInputProps()} id="pdf-upload-input" />
      <div className="upload-icon">{uploading ? '⏳' : isDragActive ? '📂' : '📄'}</div>
      <div className="upload-title">
        {uploading ? 'Processing PDF...' : isDragActive ? 'Drop it here' : 'Upload PDF'}
      </div>
      <div className="upload-sub">
        {uploading ? 'Parsing, chunking, embedding...' : 'Drag & drop or click to browse'}
      </div>
      {uploading && (
        <div className="upload-progress">
          <div className="upload-progress-bar" style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  );
}

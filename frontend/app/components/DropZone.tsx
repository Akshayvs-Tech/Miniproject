'use client';

import { useCallback, useState } from 'react';
import { Upload, Film, ImageIcon, X } from 'lucide-react';

interface DropZoneProps {
  type: 'video' | 'image';
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export default function DropZone({ type, file, onFileChange }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const accept = type === 'video' ? '.mp4,.mov,video/*' : '.jpg,.jpeg,.png,image/*';
  const title = type === 'video' ? 'Video Evidence' : 'Image Evidence';
  const supportText = type === 'video' ? 'Supports .mp4, .mov (Max 2GB)' : 'Supports .jpg, .png (High-Res preferred)';
  const Icon = type === 'video' ? Film : ImageIcon;

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) onFileChange(dropped);
    },
    [onFileChange]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) onFileChange(selected);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div
      style={{
        background: '#fff',
        borderRadius: '12px',
        padding: '1.5rem',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
        <div
          style={{
            width: '28px',
            height: '28px',
            background: '#f0ede8',
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Icon size={14} color="#1b2a4a" />
        </div>
        <span style={{ fontWeight: 600, fontSize: '0.95rem', color: '#1b2a4a' }}>{title}</span>
      </div>

      {/* Drop Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !file && document.getElementById(`file-input-${type}`)?.click()}
        style={{
          border: isDragging
            ? '2px dashed #d4a017'
            : file
            ? '2px dashed #b8a88a'
            : '2px dashed #c5c0b8',
          borderRadius: '10px',
          padding: '2rem 1.5rem',
          textAlign: 'center',
          background: isDragging ? '#fffbf0' : file ? '#faf8f4' : '#f8f7f5',
          cursor: file ? 'default' : 'pointer',
          transition: 'all 0.2s ease',
          minHeight: '180px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.75rem',
        }}
      >
        {file ? (
          <>
            {/* Uploaded — navy icon circle with gold icon, no green/red */}
            <div
              style={{
                width: '48px',
                height: '48px',
                borderRadius: '50%',
                background: '#1b2a4a',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 8px rgba(27,42,74,0.18)',
              }}
            >
              <Icon size={20} color="#d4a017" />
            </div>

            {/* File name + size */}
            <div>
              <p
                style={{
                  fontWeight: 600,
                  color: '#1b2a4a',
                  fontSize: '0.875rem',
                  maxWidth: '180px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  margin: '0 auto',
                }}
              >
                {file.name}
              </p>
              <p style={{ color: '#9ca3af', fontSize: '0.78rem', marginTop: '0.25rem' }}>
                {formatSize(file.size)}
              </p>
            </div>

            {/* Remove — neutral, no red */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onFileChange(null);
              }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.3rem 0.875rem',
                border: '1px solid #d1cfc9',
                borderRadius: '6px',
                background: '#fff',
                color: '#6b7280',
                fontSize: '0.78rem',
                cursor: 'pointer',
                fontWeight: 500,
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                const btn = e.currentTarget as HTMLButtonElement;
                btn.style.borderColor = '#1b2a4a';
                btn.style.color = '#1b2a4a';
                btn.style.background = '#f0ede8';
              }}
              onMouseLeave={(e) => {
                const btn = e.currentTarget as HTMLButtonElement;
                btn.style.borderColor = '#d1cfc9';
                btn.style.color = '#6b7280';
                btn.style.background = '#fff';
              }}
            >
              <X size={11} /> Remove
            </button>
          </>
        ) : (
          <>
            <div
              style={{
                width: '52px',
                height: '52px',
                background: '#e8e4de',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Upload size={22} color="#6b7280" />
            </div>
            <div>
              <p style={{ fontWeight: 500, color: '#374151', fontSize: '0.9rem' }}>
                Drag and drop or click to browse
              </p>
              <p style={{ color: '#9ca3af', fontSize: '0.78rem', marginTop: '0.3rem' }}>{supportText}</p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                document.getElementById(`file-input-${type}`)?.click();
              }}
              style={{
                padding: '0.4rem 1rem',
                border: '1px solid #d1cfc9',
                borderRadius: '6px',
                background: '#fff',
                color: '#374151',
                fontSize: '0.82rem',
                cursor: 'pointer',
                fontWeight: 500,
                transition: 'background 0.15s',
              }}
            >
              Browse Files
            </button>
          </>
        )}
      </div>

      <input
        id={`file-input-${type}`}
        type="file"
        accept={accept}
        onChange={handleFileInput}
        style={{ display: 'none' }}
      />
    </div>
  );
}

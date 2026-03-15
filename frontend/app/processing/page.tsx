'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAppContext } from '../lib/store';
import { analyzeEvidence } from '../lib/api';
import { useMutation } from '@tanstack/react-query';

export default function ProcessingPage() {
  const router = useRouter();
  const { uploadedVideo, uploadedImage, setAnalysisResult } = useAppContext();

  const mutation = useMutation({
    mutationFn: () => analyzeEvidence(uploadedVideo!, uploadedImage!),
    onSuccess: (result) => {
      setAnalysisResult(result);
      router.push('/summary');
    },
    onError: () => {
      router.push('/dashboard');
    },
  });

  useEffect(() => {
    if (!uploadedVideo || !uploadedImage) {
      router.push('/dashboard');
      return;
    }
    mutation.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const steps = [
    { label: 'Ingesting evidence files', done: true },
    { label: 'Analyzing video timestamps', done: mutation.isPending },
    { label: 'Processing image metadata', done: false },
    { label: 'Synthesizing AI brief', done: false },
  ];

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#f0ede8',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem',
      }}
    >
      {/* Animated Logo */}
      <div style={{ marginBottom: '2.5rem', textAlign: 'center' }}>
        <div
          style={{
            width: '80px',
            height: '80px',
            borderRadius: '50%',
            border: '3px solid #1b2a4a',
            borderTopColor: '#d4a017',
            animation: 'spin 1.2s linear infinite',
            margin: '0 auto 1.5rem',
          }}
        />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

        <h1
          style={{
            fontSize: '1.6rem',
            fontWeight: 700,
            color: '#1b2a4a',
            fontFamily: "'Playfair Display', serif",
            marginBottom: '0.5rem',
          }}
        >
          Analyzing Your Evidence
        </h1>
        <p style={{ color: '#6b7280', fontSize: '0.9rem' }}>
          Our AI is synthesizing all uploaded materials into an executive brief
        </p>
      </div>

      {/* Progress Steps */}
      <div
        style={{
          background: '#fff',
          borderRadius: '12px',
          padding: '1.75rem 2.5rem',
          boxShadow: '0 4px 24px rgba(0,0,0,0.07)',
          minWidth: '340px',
          maxWidth: '420px',
          width: '100%',
        }}
      >
        {steps.map((step, i) => (
          <ProcessingStep
            key={i}
            label={step.label}
            index={i}
            total={steps.length}
          />
        ))}
      </div>

      <p style={{ marginTop: '1.5rem', color: '#9ca3af', fontSize: '0.8rem' }}>
        This may take a few moments. Please do not close this tab.
      </p>
    </div>
  );
}

function ProcessingStep({ label, index, total }: { label: string; index: number; total: number }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.875rem',
        padding: '0.75rem 0',
        borderBottom: index < total - 1 ? '1px solid #f3f0eb' : 'none',
        animation: `fadeInUp 0.4s ease ${index * 0.15}s both`,
      }}
    >
      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
      <div
        style={{
          width: '28px',
          height: '28px',
          borderRadius: '50%',
          background: '#f0ede8',
          border: '2px solid #d4a017',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          animation: 'pulse 1.5s ease-in-out infinite',
          animationDelay: `${index * 0.3}s`,
        }}
      >
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: '#d4a017',
          }}
        />
      </div>
      <span style={{ color: '#374151', fontSize: '0.875rem', fontWeight: 500 }}>{label}</span>
    </div>
  );
}

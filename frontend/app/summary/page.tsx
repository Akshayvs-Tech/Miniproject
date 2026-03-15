'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useRef } from 'react';
import { useAppContext } from '../lib/store';
import { Download, Printer, Calendar, Shield, ArrowLeft, Video, Image as Img } from 'lucide-react';

export default function ResultsPage() {
  const router = useRouter();
  const { analysisResult } = useAppContext();
  const printRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!analysisResult) {
      router.push('/dashboard');
    }
  }, [analysisResult, router]);

  if (!analysisResult) return null;

  const handlePrint = () => {
    window.print();
  };

  const handleExportPDF = () => {
    // Trigger browser's print-to-PDF dialog
    window.print();
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f0ede8' }}>
      {/* Top Action Bar */}
      <div
        className="no-print"
        style={{
          background: '#fff',
          borderBottom: '1px solid #e8e4de',
          padding: '0.875rem 2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          position: 'sticky',
          top: 0,
          zIndex: 50,
        }}
      >
        <button
          onClick={() => router.push('/dashboard')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: 'none',
            border: 'none',
            color: '#6b7280',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontWeight: 500,
          }}
        >
          <ArrowLeft size={16} /> Back to Dashboard
        </button>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            onClick={handleExportPDF}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.5rem 1.1rem',
              border: '1px solid #d1cfc9',
              borderRadius: '8px',
              background: '#fff',
              color: '#374151',
              cursor: 'pointer',
              fontSize: '0.85rem',
              fontWeight: 500,
              transition: 'background 0.15s',
            }}
          >
            <Download size={15} /> Export PDF
          </button>
          <button
            onClick={handlePrint}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.5rem 1.1rem',
              border: '1px solid #d1cfc9',
              borderRadius: '8px',
              background: '#fff',
              color: '#374151',
              cursor: 'pointer',
              fontSize: '0.85rem',
              fontWeight: 500,
            }}
          >
            <Printer size={15} /> Print
          </button>
        </div>
      </div>

      {/* Document */}
      <div style={{ padding: '2.5rem 1rem' }}>
        <div
          ref={printRef}
          style={{
            maxWidth: '680px',
            margin: '0 auto',
            background: '#fff',
            borderRadius: '12px',
            boxShadow: '0 4px 32px rgba(0,0,0,0.08)',
            overflow: 'hidden',
          }}
        >
          {/* Document Header */}
          <div style={{ padding: '2.5rem 3rem 2rem', textAlign: 'center', borderBottom: '1px solid #f0ede8' }}>
            <div
              style={{
                width: '52px',
                height: '52px',
                borderRadius: '50%',
                border: '2px solid #d4a017',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1rem',
              }}
            >
              <Shield size={22} color="#d4a017" />
            </div>
            <p
              style={{
                fontSize: '0.7rem',
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: '#9ca3af',
                marginBottom: '0.5rem',
                fontWeight: 600,
              }}
            >
              Executive Brief
            </p>
            <h1
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: '1.9rem',
                fontWeight: 700,
                color: '#1b2a4a',
                marginBottom: '0.75rem',
              }}
            >
              Case Summary
            </h1>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.4rem',
                color: '#6b7280',
                fontSize: '0.8rem',
              }}
            >
              <Calendar size={13} />
              <span>Generated on {analysisResult.generatedDate}</span>
            </div>
          </div>

          {/* Document Body */}
          <div style={{ padding: '2rem 3rem 3rem' }}>
            {/* Section 1: Executive Overview */}
            <Section number={1} title="Executive Overview">
              <p style={{ color: '#374151', lineHeight: 1.7, fontSize: '0.9rem' }}>
                {analysisResult.executiveSummary}
              </p>
            </Section>

            {/* Section 2: Key Findings */}
            <Section number={2} title="Key Findings & Discrepancies">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {analysisResult.findings.map((finding) => (
                  <FindingCard key={finding.id} finding={finding} />
                ))}
              </div>
            </Section>

            {/* Section 3: Recommended Actions */}
            <Section number={3} title="Recommended Action Items">
              <ul style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {analysisResult.actionItems.map((item, i) => (
                  <li key={i} style={{ color: '#374151', fontSize: '0.875rem', lineHeight: 1.6 }}>
                    {item}
                  </li>
                ))}
              </ul>
            </Section>
          </div>

          {/* Document Footer */}
          <div
            style={{
              borderTop: '1px solid #f0ede8',
              padding: '1rem 3rem',
              display: 'flex',
              justifyContent: 'space-between',
              background: '#fafaf9',
            }}
          >
            <div>
              <p style={{ fontSize: '0.72rem', color: '#9ca3af' }}>Generated by Corporate Legal AI</p>
              <p style={{ fontSize: '0.72rem', color: '#9ca3af' }}>
                Chain of Custody ID: {analysisResult.chainOfCustody}
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '0.72rem', color: '#9ca3af' }}>Page 1 of 1</p>
              <p style={{ fontSize: '0.72rem', color: '#9ca3af' }}>CONFIDENTIAL WORK PRODUCT</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({ number, title, children }: { number: number; title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
        <div
          style={{
            width: '3px',
            height: '20px',
            background: '#d4a017',
            borderRadius: '2px',
            flexShrink: 0,
          }}
        />
        <h2
          style={{
            fontFamily: "'Playfair Display', serif",
            fontSize: '1.1rem',
            fontWeight: 700,
            color: '#1b2a4a',
          }}
        >
          {number}. {title}
        </h2>
      </div>
      {children}
    </div>
  );
}

function FindingCard({ finding }: { finding: { title: string; description: string; tag: string; sourceType: 'video' | 'image' } }) {
  const SourceIcon = finding.sourceType === 'video' ? Video : Img;

  return (
    <div style={{ display: 'flex', gap: '0.875rem' }}>
      <div
        style={{
          width: '20px',
          height: '20px',
          borderRadius: '50%',
          background: '#1b2a4a',
          flexShrink: 0,
          marginTop: '2px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div style={{ width: '6px', height: '6px', background: '#d4a017', borderRadius: '1px', transform: 'rotate(45deg)' }} />
      </div>
      <div style={{ flex: 1 }}>
        <h3 style={{ fontWeight: 600, fontSize: '0.9rem', color: '#1b2a4a', marginBottom: '0.35rem' }}>
          {finding.title}
        </h3>
        <p style={{ color: '#4b5563', fontSize: '0.85rem', lineHeight: 1.6, marginBottom: '0.5rem' }}>
          {finding.description}
        </p>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.3rem',
            padding: '0.2rem 0.6rem',
            background: '#f0ede8',
            borderRadius: '4px',
            fontSize: '0.72rem',
            fontFamily: 'monospace',
            fontWeight: 600,
            color: '#374151',
          }}
        >
          <SourceIcon size={11} /> {finding.tag}
        </span>
      </div>
    </div>
  );
}

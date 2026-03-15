'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAppContext } from '../lib/store';
import DropZone from '../components/DropZone';
import { Sparkles } from 'lucide-react';
import Link from 'next/link';

function getGreeting(): string {
  const now = new Date();
  const utc = now.getTime() + now.getTimezoneOffset() * 60000;
  const ist = new Date(utc + 5.5 * 3600000);
  const hour = ist.getHours();

  if (hour >= 5 && hour < 12) return 'Good Morning';
  if (hour >= 12 && hour < 17) return 'Good Afternoon';
  if (hour >= 17 && hour < 21) return 'Good Evening';
  return 'Good Evening';
}

export default function DashboardPage() {
  const router = useRouter();
  const { profile, setUploadedFiles } = useAppContext();
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const canAnalyze = videoFile !== null && imageFile !== null;
  const greeting = getGreeting();

  const firstName = profile.fullName.split(' ')[0];
  const nameParts = profile.fullName
    .replace(/,.*$/, '')
    .split(' ')
    .filter((p) => p.length > 0);
  const lastName = nameParts.length > 1 ? nameParts[nameParts.length - 1] : nameParts[0];

  const handleAnalyze = () => {
    if (!canAnalyze) return;
    setUploadedFiles(videoFile, imageFile);
    setIsLoading(true);
    router.push('/processing');
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f0ede8', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navbar */}
      <header
        style={{
          padding: '1rem 2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              background: '#1b2a4a',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <polygon points="23 7 16 12 23 17 23 7" fill="#d4a017" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" fill="white" />
            </svg>
          </div>
          <span
            style={{
              fontWeight: 700,
              fontSize: '0.85rem',
              color: '#1b2a4a',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
            }}
          >
            Track Vision
          </span>
        </div>

        {/* Avatar */}
        <Link href="/profile" style={{ display: 'flex', textDecoration: 'none' }}>
          {profile.photoUrl ? (
            <img
              src={profile.photoUrl}
              alt={firstName}
              style={{
                width: '42px',
                height: '42px',
                borderRadius: '50%',
                objectFit: 'cover',
                border: '2px solid #d4a017',
                cursor: 'pointer',
              }}
            />
          ) : (
            <div
              style={{
                width: '42px',
                height: '42px',
                borderRadius: '50%',
                background: '#1b2a4a',
                border: '2px solid #d4a017',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#d4a017',
                fontWeight: 700,
                fontSize: '1rem',
                cursor: 'pointer',
              }}
            >
              {profile.fullName.charAt(0)}
            </div>
          )}
        </Link>
      </header>

      {/* Hero Section */}
      <div style={{ padding: '1rem 2.5rem 0' }}>
        <h1
          style={{
            fontFamily: "'Playfair Display', serif",
            fontSize: '2.4rem',
            fontWeight: 700,
            color: '#1b2a4a',
            marginBottom: '0.4rem',
            lineHeight: 1.2,
          }}
        >
          {greeting}, Counselor {lastName}
        </h1>
        <p style={{ color: '#9ca3af', fontSize: '0.9rem' }}>
          Select evidence files to begin AI synthesis and briefing generation.
        </p>
        <hr style={{ marginTop: '1.25rem', border: 'none', borderTop: '1px solid #ddd9d1' }} />
      </div>

      {/* Upload Cards */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem 2.5rem',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '1.25rem',
            width: '100%',
            maxWidth: '700px',
            marginBottom: '3rem',
          }}
        >
          <DropZone type="video" file={videoFile} onFileChange={setVideoFile} />
          <DropZone type="image" file={imageFile} onFileChange={setImageFile} />
        </div>

        {/* Analyze Button */}
        <button
          onClick={handleAnalyze}
          disabled={!canAnalyze || isLoading}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.875rem 2.5rem',
            background: canAnalyze && !isLoading
              ? 'linear-gradient(135deg, #1b2a4a 0%, #243558 100%)'
              : '#9ca3af',
            color: '#fff',
            border: 'none',
            borderRadius: '10px',
            fontSize: '1rem',
            fontWeight: 600,
            cursor: canAnalyze && !isLoading ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s ease',
            boxShadow: canAnalyze && !isLoading ? '0 4px 20px rgba(27, 42, 74, 0.3)' : 'none',
          }}
          onMouseEnter={(e) => {
            if (canAnalyze && !isLoading) {
              (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(-1px)';
              (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 6px 24px rgba(27, 42, 74, 0.4)';
            }
          }}
          onMouseLeave={(e) => {
            if (canAnalyze && !isLoading) {
              (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(0)';
              (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 4px 20px rgba(27, 42, 74, 0.3)';
            }
          }}
        >
          <Sparkles size={18} />
          {isLoading ? 'Processing...' : 'Summarize Evidence'}
        </button>

        {!canAnalyze && (
          <p style={{ color: '#9ca3af', fontSize: '0.8rem', marginTop: '0.75rem' }}>
            Upload both a video and an image to enable analysis
          </p>
        )}
      </div>
    </div>
  );
}

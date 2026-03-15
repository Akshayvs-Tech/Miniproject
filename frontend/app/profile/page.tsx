'use client';

import { useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { profileSchema, ProfileFormData } from '../lib/schemas';
import { useAppContext } from '../lib/store';
import Navbar from '../components/Navbar';
import { Camera, Trash2, Info, CheckCircle } from 'lucide-react';
import Link from 'next/link';

export default function ProfilePage() {
  const { profile, setProfile } = useAppContext();
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [photoViewOpen, setPhotoViewOpen] = useState(false);
  const photoInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const {
    register,
    handleSubmit,
    formState: { errors, isDirty },
    reset,
  } = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      fullName: profile.fullName,
      jobTitle: profile.jobTitle,
      roleDepartment: profile.roleDepartment,
    },
  });

  // Simulate save mutation
  const saveMutation = useMutation({
    mutationFn: async (data: ProfileFormData) => {
      await new Promise((r) => setTimeout(r, 800));
      return data;
    },
    onSuccess: (data) => {
      setProfile({
        fullName: data.fullName,
        jobTitle: data.jobTitle,
        roleDepartment: data.roleDepartment,
      });
      reset(data);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
      queryClient.invalidateQueries({ queryKey: ['profile'] });
    },
  });

  const handlePhotoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setProfile({ photoUrl: url });
    }
  };

  const handleRemovePhoto = () => {
    setProfile({ photoUrl: '' });
    if (photoInputRef.current) photoInputRef.current.value = '';
  };

  const onSubmit = (data: ProfileFormData) => {
    saveMutation.mutate(data);
  };

  const firstName = profile.fullName.split(' ')[0];

  return (
    <div style={{ minHeight: '100vh', background: '#f4f5f7', display: 'flex', flexDirection: 'column' }}>
      <Navbar showFullNav={true} />

      <main style={{ flex: 1, padding: '2.5rem 1rem' }}>
        <div style={{ maxWidth: '620px', margin: '0 auto' }}>
          {/* Page Header */}
          <div style={{ marginBottom: '2rem' }}>
            <h1
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: '1.75rem',
                fontWeight: 700,
                color: '#1b2a4a',
                marginBottom: '0.4rem',
              }}
            >
              Personal Profile
            </h1>
            <p style={{ color: '#6b7280', fontSize: '0.9rem' }}>
              Manage your professional identity and account details.
            </p>
          </div>

          {/* Success Banner */}
          {saveSuccess && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem',
                padding: '0.75rem 1rem',
                background: '#f0fdf4',
                border: '1px solid #bbf7d0',
                borderRadius: '8px',
                marginBottom: '1rem',
                color: '#15803d',
                fontSize: '0.875rem',
                fontWeight: 500,
                animation: 'fadeInUp 0.3s ease',
              }}
            >
              <CheckCircle size={16} /> Profile saved successfully!
            </div>
          )}

          {/* Main Card */}
          <form onSubmit={handleSubmit(onSubmit)}>
            <div
              style={{
                background: '#fff',
                borderRadius: '12px',
                padding: '2rem',
                boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                marginBottom: '1.25rem',
              }}
            >
              {/* Profile Photo Section */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1.75rem',
                  paddingBottom: '1.75rem',
                  marginBottom: '1.75rem',
                  borderBottom: '1px solid #f3f0eb',
                }}
              >
                {/* Avatar — click to view full photo */}
                <div style={{ position: 'relative', flexShrink: 0 }}>
                  <div
                    onClick={() => profile.photoUrl && setPhotoViewOpen(true)}
                    style={{
                      width: '96px',
                      height: '96px',
                      borderRadius: '50%',
                      border: '3px solid #d4a017',
                      overflow: 'hidden',
                      background: '#f0ede8',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: profile.photoUrl ? 'zoom-in' : 'default',
                      transition: 'opacity 0.15s',
                    }}
                    title={profile.photoUrl ? 'Click to view photo' : ''}
                  >
                    {profile.photoUrl ? (
                      <img
                        src={profile.photoUrl}
                        alt={firstName}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    ) : (
                      <span style={{ fontSize: '2rem', fontWeight: 700, color: '#1b2a4a' }}>
                        {profile.fullName.charAt(0)}
                      </span>
                    )}
                  </div>
                </div>

                {/* Photo Info */}
                <div style={{ flex: 1 }}>
                  <h3 style={{ fontWeight: 600, color: '#1b2a4a', marginBottom: '0.3rem', fontSize: '0.95rem' }}>
                    Profile Photo
                  </h3>
                  <p style={{ color: '#6b7280', fontSize: '0.8rem', marginBottom: '0.875rem', lineHeight: 1.5 }}>
                    Upload a professional headshot. Recommended size: 400×400px, JPG or PNG.
                  </p>
                  <div style={{ display: 'flex', gap: '0.625rem' }}>
                    <button
                      type="button"
                      onClick={() => photoInputRef.current?.click()}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                        padding: '0.45rem 0.9rem',
                        background: '#1b2a4a',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '7px',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        letterSpacing: '0.02em',
                      }}
                    >
                      <Camera size={13} /> CHANGE PHOTO
                    </button>
                    <button
                      type="button"
                      onClick={handleRemovePhoto}
                      style={{
                        padding: '0.45rem 0.9rem',
                        background: '#fff',
                        color: '#374151',
                        border: '1px solid #d1cfc9',
                        borderRadius: '7px',
                        fontSize: '0.8rem',
                        fontWeight: 500,
                        cursor: 'pointer',
                      }}
                    >
                      REMOVE
                    </button>
                  </div>
                  <input
                    ref={photoInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handlePhotoChange}
                    style={{ display: 'none' }}
                  />
                </div>
              </div>

              {/* Form Fields */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {/* Full Name */}
                <div>
                  <label style={labelStyle}>FULL LEGAL NAME</label>
                  <input
                    {...register('fullName')}
                    style={{
                      ...inputStyle,
                      borderColor: errors.fullName ? '#fca5a5' : '#d1cfc9',
                    }}
                    placeholder="Jonathan H. Sterling, Esq."
                  />
                  {errors.fullName && (
                    <p style={{ color: '#dc2626', fontSize: '0.75rem', marginTop: '0.3rem' }}>
                      {errors.fullName.message}
                    </p>
                  )}
                </div>

                {/* Job Title + Role */}
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <div style={{ flex: 1 }}>
                    <label style={labelStyle}>JOB TITLE</label>
                    <input
                      {...register('jobTitle')}
                      style={{
                        ...inputStyle,
                        borderColor: errors.jobTitle ? '#fca5a5' : '#d1cfc9',
                      }}
                      placeholder="Senior Partner"
                    />
                    {errors.jobTitle && (
                      <p style={{ color: '#dc2626', fontSize: '0.75rem', marginTop: '0.3rem' }}>
                        {errors.jobTitle.message}
                      </p>
                    )}
                  </div>
                  <div style={{ flex: 1 }}>
                    <label style={labelStyle}>ROLE / DEPARTMENT</label>
                    <input
                      {...register('roleDepartment')}
                      style={{
                        ...inputStyle,
                        borderColor: errors.roleDepartment ? '#fca5a5' : '#d1cfc9',
                      }}
                      placeholder="Corporate Law & Litigation"
                    />
                    {errors.roleDepartment && (
                      <p style={{ color: '#dc2626', fontSize: '0.75rem', marginTop: '0.3rem' }}>
                        {errors.roleDepartment.message}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Form Actions */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: '0.75rem',
                  marginTop: '2rem',
                  paddingTop: '1.5rem',
                  borderTop: '1px solid #f3f0eb',
                }}
              >
                <button
                  type="button"
                  onClick={() => reset()}
                  style={{
                    padding: '0.6rem 1.5rem',
                    background: '#fff',
                    color: '#374151',
                    border: '1px solid #d1cfc9',
                    borderRadius: '8px',
                    fontSize: '0.875rem',
                    fontWeight: 500,
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saveMutation.isPending}
                  style={{
                    padding: '0.6rem 1.75rem',
                    background: saveMutation.isPending ? '#6b7280' : '#1b2a4a',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '0.875rem',
                    fontWeight: 600,
                    cursor: saveMutation.isPending ? 'not-allowed' : 'pointer',
                    transition: 'background 0.2s',
                    minWidth: '130px',
                  }}
                >
                  {saveMutation.isPending ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            </div>
          </form>

          {/* Notice Card */}
          <div
            style={{
              background: '#fff',
              borderRadius: '10px',
              padding: '1rem 1.25rem',
              display: 'flex',
              gap: '0.75rem',
              alignItems: 'flex-start',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
            }}
          >
            <Info size={16} color="#3b82f6" style={{ flexShrink: 0, marginTop: '1px' }} />
            <p style={{ color: '#374151', fontSize: '0.8rem', lineHeight: 1.6 }}>
              <strong>Note:</strong> Official Bar ID and Firm Affiliation updates require secondary verification from the
              Administrative Office. To request changes to professional credentials, please contact the Compliance Department.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          background: '#1b2a4a',
          color: '#fff',
          padding: '1.5rem 2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div
            style={{
              width: '28px',
              height: '28px',
              background: '#d4a017',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <polygon points="23 7 16 12 23 17 23 7" fill="white" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" fill="white" />
            </svg>
          </div>
          <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Track Vision</span>
        </div>

        <p style={{ color: '#9ca3af', fontSize: '0.78rem' }}>
          © 2023 Track Vision. All Rights Reserved. Encrypted Legal Professional Portal.
        </p>

        <div style={{ display: 'flex', gap: '1.5rem' }}>
          {['PRIVACY', 'TERMS', 'ETHICS'].map((link) => (
            <a
              key={link}
              href="#"
              style={{
                color: '#9ca3af',
                fontSize: '0.78rem',
                fontWeight: 600,
                textDecoration: 'none',
                letterSpacing: '0.05em',
              }}
            >
              {link}
            </a>
          ))}
        </div>
      </footer>
      {/* Photo Lightbox Modal */}
      {photoViewOpen && profile.photoUrl && (
        <div
          onClick={() => setPhotoViewOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            cursor: 'zoom-out',
            animation: 'fadeInUp 0.2s ease',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              position: 'relative',
              maxWidth: '90vw',
              maxHeight: '90vh',
              borderRadius: '16px',
              overflow: 'hidden',
              boxShadow: '0 24px 80px rgba(0,0,0,0.5)',
            }}
          >
            <img
              src={profile.photoUrl}
              alt={firstName}
              style={
                { maxWidth: '80vw', maxHeight: '80vh', display: 'block', objectFit: 'contain' }
              }
            />
            <button
              onClick={() => setPhotoViewOpen(false)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '12px',
                background: 'rgba(0,0,0,0.5)',
                color: '#fff',
                border: 'none',
                borderRadius: '50%',
                width: '32px',
                height: '32px',
                fontSize: '1.1rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                lineHeight: 1,
              }}
            >
              ✕
            </button>
          </div>
          <p
            style={{
              position: 'absolute',
              bottom: '1.5rem',
              color: 'rgba(255,255,255,0.6)',
              fontSize: '0.8rem',
            }}
          >
            Click anywhere outside to close
          </p>
        </div>
      )}
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: '0.7rem',
  fontWeight: 700,
  color: '#6b7280',
  letterSpacing: '0.08em',
  marginBottom: '0.4rem',
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '0.6rem 0.875rem',
  border: '1px solid #d1cfc9',
  borderRadius: '8px',
  fontSize: '0.9rem',
  color: '#1b2a4a',
  outline: 'none',
  background: '#fff',
  fontFamily: 'inherit',
  transition: 'border-color 0.15s',
};

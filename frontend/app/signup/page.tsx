'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation } from '@tanstack/react-query';
import { signupSchema, SignupFormData } from '../lib/schemas';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { User, Mail, Phone, Lock, ArrowRight, Eye, EyeOff } from 'lucide-react';

export default function SignupPage() {
  const router = useRouter();
  const [showPassword, setShowPassword] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<SignupFormData>({
    resolver: zodResolver(signupSchema),
    defaultValues: { agreed: undefined },
  });

  const signupMutation = useMutation({
    mutationFn: async (data: SignupFormData) => {
      await new Promise((r) => setTimeout(r, 1200));
      return data;
    },
    onSuccess: () => {
      router.push('/login');
    },
  });

  const onSubmit = (data: SignupFormData) => signupMutation.mutate(data);

  return (
    <div style={{ minHeight: '100vh', background: '#eeeae4', display: 'flex', flexDirection: 'column' }}>
      {/* Top Nav */}
      <header
        style={{
          background: '#fff',
          borderBottom: '1px solid #e8e4de',
          padding: '0 2rem',
          height: '56px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div
            style={{
              width: '28px',
              height: '28px',
              background: '#1b2a4a',
              borderRadius: '5px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <polygon points="23 7 16 12 23 17 23 7" fill="#d4a017" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" fill="white" />
            </svg>
          </div>
          <span style={{ fontWeight: 700, fontSize: '0.85rem', color: '#1b2a4a', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            Track Vision
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
          <span style={{ color: '#6b7280', fontSize: '0.875rem', cursor: 'pointer' }}>Support</span>
          <Link
            href="/login"
            style={{
              padding: '0.4rem 1.1rem',
              border: '1.5px solid #1b2a4a',
              borderRadius: '7px',
              color: '#1b2a4a',
              fontSize: '0.85rem',
              fontWeight: 600,
              textDecoration: 'none',
              transition: 'background 0.15s',
            }}
          >
            Login
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2.5rem 1rem',
        }}
      >
        <div
          style={{
            display: 'flex',
            width: '100%',
            maxWidth: '900px',
            borderRadius: '16px',
            overflow: 'hidden',
            boxShadow: '0 8px 40px rgba(0,0,0,0.12)',
          }}
        >
          {/* Left Panel */}
          <div
            style={{
              flex: '0 0 42%',
              background: 'linear-gradient(160deg, #1b2a4a 0%, #243558 100%)',
              padding: '3rem 2.5rem',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            {/* Decorative BG circle */}
            <div
              style={{
                position: 'absolute',
                bottom: '-80px',
                right: '-80px',
                width: '280px',
                height: '280px',
                borderRadius: '50%',
                background: 'rgba(212,160,23,0.06)',
              }}
            />

            <div>
              {/* Icon */}
              <div
                style={{
                  width: '48px',
                  height: '48px',
                  background: '#d4a017',
                  borderRadius: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '2.5rem',
                }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <polyline points="9 22 9 12 15 12 15 22" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>

              <h2
                style={{
                  fontFamily: "'Playfair Display', serif",
                  fontSize: '2rem',
                  fontWeight: 700,
                  color: '#fff',
                  lineHeight: 1.25,
                  marginBottom: '1.25rem',
                }}
              >
                Securing the future of legal operations.
              </h2>
              <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.9rem', lineHeight: 1.65 }}>
                Access the premier suite for corporate governance, compliance tracking, and litigation management.
              </p>
            </div>

            {/* Bottom user card */}
            <div
              style={{
                borderTop: '1px solid rgba(255,255,255,0.12)',
                paddingTop: '1.5rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.875rem',
              }}
            >
              <div
                style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: '#1b2a4a',
                  border: '2px solid #d4a017',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  overflow: 'hidden',
                }}
              >
                <span style={{ color: '#d4a017', fontWeight: 700, fontSize: '0.9rem' }}>J</span>
              </div>
              <div>
                <p style={{ color: '#fff', fontWeight: 600, fontSize: '0.875rem' }}>Jonathan Sterling</p>
                <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.75rem' }}>Senior Partner, Sterling &amp; Associates</p>
              </div>
            </div>
          </div>

          {/* Right Panel — Form */}
          <div style={{ flex: 1, background: '#fff', padding: '3rem 2.75rem' }}>
            <h1
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: '1.75rem',
                fontWeight: 700,
                color: '#1b2a4a',
                marginBottom: '0.35rem',
              }}
            >
              Join Track Vision
            </h1>
            <p style={{ color: '#9ca3af', fontSize: '0.85rem', marginBottom: '2rem' }}>
              Professional registration for the Legal Dashboard.
            </p>

            <form onSubmit={handleSubmit(onSubmit)} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
              {/* Full Name */}
              <FormField
                label="FULL NAME"
                error={errors.fullName?.message}
                icon={<User size={14} color="#9ca3af" />}
              >
                <input
                  {...register('fullName')}
                  placeholder="e.g. Julian Vane"
                  style={inputStyle(!!errors.fullName)}
                />
              </FormField>

              {/* Work Email */}
              <FormField
                label="WORK EMAIL ADDRESS"
                error={errors.workEmail?.message}
                icon={<Mail size={14} color="#9ca3af" />}
              >
                <input
                  {...register('workEmail')}
                  type="email"
                  placeholder="name@firm.com"
                  style={inputStyle(!!errors.workEmail)}
                />
              </FormField>

              {/* Phone */}
              <FormField
                label="PHONE NUMBER"
                error={errors.phoneNumber?.message}
                icon={<Phone size={14} color="#9ca3af" />}
              >
                <input
                  {...register('phoneNumber')}
                  type="tel"
                  placeholder="+91 98765 4XXXX"
                  style={inputStyle(!!errors.phoneNumber)}
                />
              </FormField>

              {/* Password */}
              <FormField
                label="SECURE PASSWORD"
                error={errors.password?.message}
                icon={<Lock size={14} color="#9ca3af" />}
                suffix={
                  <button
                    type="button"
                    onClick={() => setShowPassword((p) => !p)}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#9ca3af', display: 'flex', alignItems: 'center' }}
                  >
                    {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                }
              >
                <input
                  {...register('password')}
                  type={showPassword ? 'text' : 'password'}
                  placeholder="••••••••"
                  style={inputStyle(!!errors.password)}
                />
              </FormField>

              {/* Terms */}
              <div>
                <label style={{ display: 'flex', alignItems: 'flex-start', gap: '0.625rem', cursor: 'pointer' }}>
                  <input
                    {...register('agreed')}
                    type="checkbox"
                    style={{ marginTop: '2px', accentColor: '#1b2a4a', width: '14px', height: '14px', flexShrink: 0 }}
                  />
                  <span style={{ fontSize: '0.8rem', color: '#6b7280', lineHeight: 1.5 }}>
                    I certify that I am a licensed legal professional and agree to the{' '}
                    <a href="#" style={{ color: '#d4a017', fontWeight: 600, textDecoration: 'none' }}>
                      Terms of Confidentiality
                    </a>
                    .
                  </span>
                </label>
                {errors.agreed && (
                  <p style={{ color: '#dc2626', fontSize: '0.72rem', marginTop: '0.25rem', marginLeft: '1.4rem' }}>
                    {errors.agreed.message as string}
                  </p>
                )}
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={signupMutation.isPending}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.5rem',
                  width: '100%',
                  padding: '0.875rem',
                  background: signupMutation.isPending ? '#6b7280' : '#1b2a4a',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '0.85rem',
                  fontWeight: 700,
                  letterSpacing: '0.08em',
                  cursor: signupMutation.isPending ? 'not-allowed' : 'pointer',
                  transition: 'background 0.2s',
                  marginTop: '0.25rem',
                }}
              >
                {signupMutation.isPending ? 'CREATING ACCOUNT...' : 'INITIALIZE ACCOUNT'}
                {!signupMutation.isPending && <ArrowRight size={16} />}
              </button>
            </form>

            <p style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.85rem', color: '#6b7280' }}>
              Already a member?{' '}
              <Link href="/login" style={{ color: '#d4a017', fontWeight: 700, textDecoration: 'none' }}>
                Access Dashboard
              </Link>
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          padding: '1rem 2.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}
      >
        <div style={{ display: 'flex', gap: '1.75rem' }}>
          {['PRIVACY PROTOCOL', 'GOVERNANCE STANDARDS', 'SYSTEM STATUS'].map((t) => (
            <a key={t} href="#" style={{ color: '#9ca3af', fontSize: '0.7rem', letterSpacing: '0.06em', textDecoration: 'none' }}>
              {t}
            </a>
          ))}
        </div>
        <p style={{ color: '#9ca3af', fontSize: '0.7rem', letterSpacing: '0.04em' }}>
          © 2024 TRACK VISION SYSTEMS, INC. ALL RIGHTS RESERVED.
        </p>
      </footer>
    </div>
  );
}

function FormField({
  label,
  error,
  icon,
  suffix,
  children,
}: {
  label: string;
  error?: string;
  icon: React.ReactNode;
  suffix?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label style={{ display: 'block', fontSize: '0.68rem', fontWeight: 700, color: '#6b7280', letterSpacing: '0.1em', marginBottom: '0.4rem' }}>
        {label}
      </label>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          border: `1px solid ${error ? '#fca5a5' : '#d1cfc9'}`,
          borderRadius: '8px',
          padding: '0 0.75rem',
          background: '#fff',
          transition: 'border-color 0.15s',
        }}
      >
        <span style={{ flexShrink: 0, display: 'flex' }}>{icon}</span>
        <div style={{ flex: 1 }}>{children}</div>
        {suffix && <span style={{ flexShrink: 0, display: 'flex' }}>{suffix}</span>}
      </div>
      {error && <p style={{ color: '#dc2626', fontSize: '0.72rem', marginTop: '0.25rem' }}>{error}</p>}
    </div>
  );
}

function inputStyle(hasError: boolean): React.CSSProperties {
  return {
    width: '100%',
    padding: '0.65rem 0',
    border: 'none',
    outline: 'none',
    fontSize: '0.875rem',
    color: '#1b2a4a',
    background: 'transparent',
    fontFamily: 'inherit',
  };
}

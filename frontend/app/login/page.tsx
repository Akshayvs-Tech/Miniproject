'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation } from '@tanstack/react-query';
import { loginSchema, LoginFormData } from '../lib/schemas';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Mail, Phone, Lock, ArrowRight, Eye, EyeOff, ShieldCheck } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const [showPassword, setShowPassword] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  const loginMutation = useMutation({
    mutationFn: async (data: LoginFormData) => {
      await new Promise((r) => setTimeout(r, 1200));
      return data;
    },
    onSuccess: () => {
      router.push('/dashboard');
    },
  });

  const onSubmit = (data: LoginFormData) => loginMutation.mutate(data);

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#eeeae4',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Top Nav */}
      <header
        style={{
          padding: '0 2.5rem',
          height: '56px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #ddd9d1',
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
          <span style={{ fontWeight: 700, fontSize: '0.85rem', color: '#1b2a4a', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Track Vision
          </span>
        </div>
        <span style={{ fontSize: '0.72rem', fontWeight: 600, color: '#9ca3af', letterSpacing: '0.12em' }}>
          COUNSEL PORTAL V4.2
        </span>
      </header>

      {/* Scales of justice watermark */}
      <div
        style={{
          position: 'fixed',
          bottom: '2rem',
          left: '2.5rem',
          opacity: 0.1,
          pointerEvents: 'none',
        }}
      >
        <svg width="100" height="100" viewBox="0 0 24 24" fill="#1b2a4a">
          <path d="M12 2L3 7l1.5 9H12m0-14l9 5-1.5 9H12m0 0v4M8 21h8" stroke="#1b2a4a" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
          <circle cx="12" cy="2" r="1" fill="#1b2a4a"/>
          <line x1="4.5" y1="16" x2="11" y2="16" stroke="#1b2a4a" strokeWidth="1.5"/>
          <line x1="13" y1="16" x2="19.5" y2="16" stroke="#1b2a4a" strokeWidth="1.5"/>
        </svg>
      </div>

      {/* Main */}
      <main
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem 1rem',
        }}
      >
        <div
          style={{
            background: '#fff',
            borderRadius: '16px',
            padding: '3rem 2.75rem 2.5rem',
            width: '100%',
            maxWidth: '440px',
            boxShadow: '0 8px 40px rgba(0,0,0,0.1)',
          }}
        >
          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: '2.25rem' }}>
            <h1
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: '2rem',
                fontWeight: 700,
                color: '#1b2a4a',
                lineHeight: 1.25,
                marginBottom: '0.5rem',
              }}
            >
              Secure Counselor Access
            </h1>
            <p style={{ color: '#9ca3af', fontSize: '0.85rem', fontStyle: 'italic' }}>
              Authorized Legal Personnel Only
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit(onSubmit)} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
            {/* Email */}
            <LoginField
              label="PROFESSIONAL EMAIL"
              error={errors.email?.message}
              icon={<Mail size={14} color="#9ca3af" />}
            >
              <input
                {...register('email')}
                type="email"
                placeholder="e.g. counsel@firm.com"
                style={inputStyle}
              />
            </LoginField>

            {/* Phone */}
            <LoginField
              label="PHONE NUMBER"
              error={errors.phoneNumber?.message}
              icon={<Phone size={14} color="#9ca3af" />}
            >
              <input
                {...register('phoneNumber')}
                type="tel"
                placeholder="e.g. +91 98765 4XXXX"
                style={inputStyle}
              />
            </LoginField>

            {/* Password */}
            <LoginField
              label="SECURITY PASSWORD"
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
                style={inputStyle}
              />
            </LoginField>

            {/* Remember + Forgot */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                <input
                  {...register('remember')}
                  type="checkbox"
                  style={{ accentColor: '#1b2a4a', width: '13px', height: '13px' }}
                />
                <span style={{ fontSize: '0.82rem', color: '#6b7280' }}>Remember credentials</span>
              </label>
              <a
                href="#"
                style={{
                  fontSize: '0.82rem',
                  color: '#1b2a4a',
                  fontWeight: 700,
                  fontStyle: 'italic',
                  textDecoration: 'none',
                  fontFamily: "'Playfair Display', serif",
                }}
              >
                Forgot Password?
              </a>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loginMutation.isPending}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                width: '100%',
                padding: '0.9rem',
                background: loginMutation.isPending ? '#6b7280' : '#1b2a4a',
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: 700,
                letterSpacing: '0.08em',
                cursor: loginMutation.isPending ? 'not-allowed' : 'pointer',
                transition: 'background 0.2s',
                marginTop: '0.25rem',
              }}
            >
              {loginMutation.isPending ? 'AUTHENTICATING...' : 'AUTHENTICATE ACCESS'}
              {!loginMutation.isPending && <ArrowRight size={16} />}
            </button>
          </form>

          {/* Security Badges */}
          <div style={{ marginTop: '2rem', borderTop: '1px solid #f0ede8', paddingTop: '1.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '2.5rem', marginBottom: '1rem' }}>
              {[
                { label: 'ENCRYPTED' },
                { label: 'MONITORED' },
                { label: 'COMPLIANT' },
              ].map(({ label }) => (
                <div key={label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.35rem' }}>
                  <ShieldCheck size={22} color="#c5c0b8" />
                  <span style={{ fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.1em', color: '#9ca3af' }}>
                    {label}
                  </span>
                </div>
              ))}
            </div>
            <p style={{ textAlign: 'center', color: '#9ca3af', fontSize: '0.75rem', lineHeight: 1.6 }}>
              This system is for the use of authorized users only. All activities are logged and monitored for security compliance.
            </p>
          </div>

          {/* Sign up link */}
          <p style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: '0.82rem', color: '#6b7280' }}>
            New to Track Vision?{' '}
            <Link href="/signup" style={{ color: '#d4a017', fontWeight: 700, textDecoration: 'none' }}>
              Create Account
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}

function LoginField({
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
          borderRadius: '7px',
          padding: '0 0.75rem',
          background: '#fff',
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

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '0.65rem 0',
  border: 'none',
  outline: 'none',
  fontSize: '0.875rem',
  color: '#1b2a4a',
  background: 'transparent',
  fontFamily: 'inherit',
};

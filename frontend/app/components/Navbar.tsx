'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAppContext } from '../lib/store';
import { LogOut, User } from 'lucide-react';

interface NavbarProps {
  showFullNav?: boolean;
}

export default function Navbar({ showFullNav = true }: NavbarProps) {
  const pathname = usePathname();
  const { profile } = useAppContext();

  const firstName = profile.fullName.split(' ')[0];

  return (
    <nav
      style={{
        background: '#fff',
        borderBottom: '1px solid #e8e4de',
        padding: '0 2rem',
        height: '64px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}
    >
      {/* Logo */}
      <Link href="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none' }}>
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
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <polygon points="23 7 16 12 23 17 23 7" fill="#d4a017" />
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2" fill="white" />
          </svg>
        </div>
        <span style={{ fontWeight: 700, fontSize: '1rem', color: '#1b2a4a', letterSpacing: '-0.01em' }}>
          Track Vision
        </span>
      </Link>

      {/* Nav Links */}
      {showFullNav && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <NavLink href="/dashboard" label="Dashboard" active={pathname === '/dashboard'} />
          <NavLink href="/profile" label="Profile" active={pathname === '/profile'} />
        </div>
      )}

      {/* Right side: Avatar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <Link href="/profile" style={{ display: 'flex', textDecoration: 'none' }}>
          {profile.photoUrl ? (
            <img
              src={profile.photoUrl}
              alt={firstName}
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                objectFit: 'cover',
                border: '2px solid #d4a017',
              }}
            />
          ) : (
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                background: '#1b2a4a',
                border: '2px solid #d4a017',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '0.8rem',
                fontWeight: 600,
              }}
            >
              {profile.fullName.charAt(0)}
            </div>
          )}
        </Link>
      </div>
    </nav>
  );
}

function NavLink({ href, label, active }: { href: string; label: string; active: boolean }) {
  return (
    <Link
      href={href}
      style={{
        padding: '0.375rem 0.875rem',
        borderRadius: '6px',
        fontSize: '0.875rem',
        fontWeight: active ? 600 : 400,
        color: active ? '#1b2a4a' : '#6b7280',
        textDecoration: 'none',
        borderBottom: active ? '2px solid #d4a017' : '2px solid transparent',
        transition: 'all 0.15s ease',
      }}
    >
      {label}
    </Link>
  );
}

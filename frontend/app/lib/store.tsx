'use client';

import { createContext, useContext, useState, ReactNode } from 'react';
import { AnalysisResult } from './schemas';

// ─── Types ────────────────────────────────────────────────────────────────────

interface ProfileData {
  fullName: string;
  jobTitle: string;
  roleDepartment: string;
  photoUrl: string;
}

interface AuthUser {
  id: string;
  email: string;
  fullName: string;
}

interface AppState {
  // Profile (editable on Profile page)
  profile: ProfileData;
  setProfile: (data: Partial<ProfileData>) => void;

  // Auth
  authUser: AuthUser | null;
  token: string | null;
  setAuth: (user: AuthUser, token: string) => void;
  clearAuth: () => void;

  // Analysis
  analysisResult: AnalysisResult | null;
  setAnalysisResult: (result: AnalysisResult | null) => void;

  // Uploaded files
  uploadedVideo: File | null;
  uploadedImage: File | null;
  setUploadedFiles: (video: File | null, image: File | null) => void;
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

const defaultProfile: ProfileData = {
  fullName: '',
  jobTitle: '',
  roleDepartment: '',
  photoUrl: '',
};

// ─── Context ──────────────────────────────────────────────────────────────────

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [profile, setProfileState] = useState<ProfileData>(defaultProfile);
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);

  const setProfile = (data: Partial<ProfileData>) => {
    setProfileState((prev) => ({ ...prev, ...data }));
  };

  /** Called after a successful login — stores the JWT and hydrates profile from backend user */
  const setAuth = (user: AuthUser, jwt: string) => {
    setAuthUser(user);
    setToken(jwt);
    // Pre-fill profile with the name registered during signup
    setProfileState((prev) => ({
      ...prev,
      fullName: user.fullName || prev.fullName,
    }));
  };

  /** Called on logout */
  const clearAuth = () => {
    setAuthUser(null);
    setToken(null);
  };

  const setUploadedFiles = (video: File | null, image: File | null) => {
    setUploadedVideo(video);
    setUploadedImage(image);
  };

  return (
    <AppContext.Provider
      value={{
        profile,
        setProfile,
        authUser,
        token,
        setAuth,
        clearAuth,
        analysisResult,
        setAnalysisResult,
        uploadedVideo,
        uploadedImage,
        setUploadedFiles,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppContext must be used within AppProvider');
  return ctx;
}

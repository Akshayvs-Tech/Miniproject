'use client';

import { createContext, useContext, useState, ReactNode } from 'react';
import { AnalysisResult } from './schemas';

export interface RegisteredUser {
  fullName: string;
  workEmail: string;
  phoneNumber: string;
  password: string;
}

interface ProfileData {
  fullName: string;
  jobTitle: string;
  roleDepartment: string;
  photoUrl: string;
}

interface AppState {
  profile: ProfileData;
  setProfile: (data: Partial<ProfileData>) => void;
  analysisResult: AnalysisResult | null;
  setAnalysisResult: (result: AnalysisResult | null) => void;
  uploadedVideo: File | null;
  uploadedImage: File | null;
  setUploadedFiles: (video: File | null, image: File | null) => void;
  registeredUser: RegisteredUser | null;
  registerUser: (user: RegisteredUser) => void;
  loginUser: (email: string, phone: string, password: string) => { success: boolean; error?: string };
}

const defaultProfile: ProfileData = {
  fullName: '',
  jobTitle: '',
  roleDepartment: '',
  photoUrl: '',
};

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [profile, setProfileState] = useState<ProfileData>(defaultProfile);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [registeredUser, setRegisteredUser] = useState<RegisteredUser | null>(null);

  const setProfile = (data: Partial<ProfileData>) => {
    setProfileState((prev) => ({ ...prev, ...data }));
  };

  const setUploadedFiles = (video: File | null, image: File | null) => {
    setUploadedVideo(video);
    setUploadedImage(image);
  };

  const registerUser = (user: RegisteredUser) => {
    setRegisteredUser(user);
    setProfileState((prev) => ({ ...prev, fullName: user.fullName }));
  };

  const loginUser = (email: string, phone: string, password: string): { success: boolean; error?: string } => {
    if (!registeredUser) {
      return { success: false, error: 'No account found. Please sign up first.' };
    }
    if (registeredUser.workEmail !== email) {
      return { success: false, error: 'Email address does not match our records.' };
    }
    if (registeredUser.phoneNumber !== phone) {
      return { success: false, error: 'Phone number does not match our records.' };
    }
    if (registeredUser.password !== password) {
      return { success: false, error: 'Incorrect password. Please try again.' };
    }
    return { success: true };
  };

  return (
    <AppContext.Provider
      value={{
        profile,
        setProfile,
        analysisResult,
        setAnalysisResult,
        uploadedVideo,
        uploadedImage,
        setUploadedFiles,
        registeredUser,
        registerUser,
        loginUser,
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
